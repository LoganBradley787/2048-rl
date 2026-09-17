"""Evaluate n-tuple weights at several expectimax depths and write a JSON the
report builder reads.

    uv run python examples/evaluate_ntuple.py --games 100 --depths 0,1,2,3 --json checkpoints/ntuple/evals.json

Cool mode (the agent places every tile; one game is the canonical line, so 2
games suffice): --choose evaluates the max-max tree at --depths and beams given
as width:depth[:spread[:stride]]:

    uv run python examples/evaluate_ntuple.py --weights checkpoints/ntuple_choose/weights.bin --choose --games 2 --depths 3 --beam 32:12,64:12:4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

from game2048 import ntuple as nt


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=None, help="defaults to checkpoints/ntuple/weights.bin or latest.bin")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--depths", default="0,1,2,3")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--cutoff", type=float, default=0.0,
                   help="skip spawn branches whose cumulative probability is below this (0 = exact)")
    p.add_argument("--json", default=None)
    p.add_argument("--choose", action="store_true", help="evaluate the choose game (cool mode) instead of random spawns")
    p.add_argument("--beam", default="", help="choose mode: comma-separated beams width:depth[:spread[:stride]]")
    p.add_argument("--max-moves", type=int, default=200_000, help="choose mode: cap per game")
    args = p.parse_args(argv)

    path = nt.resolve_weights(args.weights)
    net = nt.NTupleNet.load(path)
    out_dir = path.parent
    state = json.loads((out_dir / "state.json").read_text()) if (out_dir / "state.json").exists() else {}
    mps = None
    if (out_dir / "log.csv").exists():
        with open(out_dir / "log.csv", newline="") as f:
            rates = [float(r["moves_per_s"]) for r in csv.DictReader(f)]
        mps = sum(rates) / len(rates) if rates else None
    print(f"weights: {path} ({state.get('games', '?')} training games, boundaries {net.boundaries})")

    def show(label: str, stats: dict, t: float) -> None:
        print(f"{label}: {args.games} games in {time.time() - t:.0f}s | mean {stats['mean_score']:.0f}, "
              f"median {stats['median_score']:.0f}, max {stats['max_score']} | {stats['mean_length']:.0f} moves | "
              f"8192 {stats['rate_8192']:.0%}, 16384 {stats['rate_16384']:.0%}, 32768 {stats['rate_32768']:.0%}", flush=True)

    evals = {}
    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    if args.choose:
        for d in depths:
            t = time.time()
            stats = nt.summarize(net.play_choose(args.games, depth=d, topk=4, threads=args.threads, seed=args.seed,
                                                 max_moves=args.max_moves))
            evals[f"tree-{d}"] = stats
            show(f"tree-{d}", stats, t)
        for spec in [x.strip() for x in args.beam.split(",") if x.strip()]:
            parts = [int(x) for x in spec.split(":")]
            width, depth = parts[0], parts[1]
            spread = parts[2] if len(parts) > 2 else 0
            stride = parts[3] if len(parts) > 3 else 1
            t = time.time()
            stats = nt.summarize(net.play_beam(args.games, width=width, depth=depth, spread=spread, stride=stride,
                                               threads=args.threads, seed=args.seed, max_moves=args.max_moves))
            evals[f"beam-{spec}"] = stats
            show(f"beam-{spec}", stats, t)
    else:
        for d in depths:
            t = time.time()
            stats = nt.summarize(net.play(args.games, depth=d, cutoff=args.cutoff, threads=args.threads, seed=args.seed))
            evals[d] = stats
            show(f"depth {d}", stats, t)
    if args.json:
        meta = {**state, "threads": state.get("threads", args.threads), "eval_threads": args.threads,
                "cutoff": args.cutoff, "moves_per_s": mps, "weights": str(path),
                "mode": "choose" if args.choose else "random"}
        Path(args.json).write_text(json.dumps({"evals": evals, "meta": meta}, indent=1))
        print(f"wrote {args.json}")
    net.close()


if __name__ == "__main__":
    main()
