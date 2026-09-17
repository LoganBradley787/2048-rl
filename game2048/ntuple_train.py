"""Train the n-tuple network with TD(0) self-play in native code.

    uv run python -m game2048.ntuple_train --hours 8 --threads 12

Writes to --out: latest.bin (weights + learning state), weights.bin (weights
only, what the UI loads), state.json (counters), log.csv (one row per chunk
of training games) and eval.csv (expectimax evaluations). Resume with
--resume <out>/latest.bin.

--choose trains for cool mode (the player places every tile): self-play uses
the choose search and evaluations play the choose game. Keep --max-moves far
above the natural game length; games cut at the cap leave their late boards
ungrounded, values inflate and the policy collapses.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

from . import ntuple as nt

LOG_FIELDS = [
    "games", "chunk_moves", "moves", "elapsed_s", "moves_per_s", "mean_score", "median_score",
    "max_score", "mean_length", "max_tile", "rate_2048", "rate_4096", "rate_8192", "rate_16384",
    "rate_32768",
]
EVAL_FIELDS = [
    "games", "elapsed_s", "mode", "depth", "eval_games", "mean_score", "median_score", "max_score",
    "mean_length", "max_tile", "rate_2048", "rate_4096", "rate_8192", "rate_16384", "rate_32768",
]


def _writer(path: Path, fields: list[str], append: bool):
    if append and path.exists():
        with open(path, newline="") as f:
            header = next(csv.reader(f), None)
        if header:
            fields = header  # a file from an older run keeps its own columns
    f = open(path, "a" if append else "w", newline="")
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    if not append:
        w.writeheader()
    return f, w


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", type=int, default=None, help="training games to play this run")
    p.add_argument("--hours", type=float, default=None, help="wall-clock budget for this run")
    p.add_argument("--chunk", type=int, default=5000, help="games per native call and per log row")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    p.add_argument("--alpha", type=float, default=1.0, help="step for temporal-coherence stages")
    p.add_argument("--alpha-plain", type=float, default=0.025,
                   help="total step for plain-TD stages (only used with --no-tc or --tc-stages)")
    p.add_argument("--tc-stages", type=int, default=None,
                   help="leading stages that keep temporal-coherence state (default: all, or 0 with --no-tc)")
    p.add_argument("--no-tc", action="store_true",
                   help="plain TD instead of temporal coherence (with --resume: drop the TC state, a third of the memory)")
    p.add_argument("--stages", default="", help="comma-separated tile-mass boundaries for multi-stage tables, e.g. 16384,24576,32768")
    p.add_argument("--patterns", default="4x6", choices=sorted(nt.PATTERN_SETS), help="pattern set for a new network")
    p.add_argument("--out", default="checkpoints/ntuple")
    p.add_argument("--resume", default=None, help="weights file to continue from")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every-min", type=float, default=10.0, help="0 = only at the end")
    p.add_argument("--eval-every-min", type=float, default=30.0, help="0 = never")
    p.add_argument("--eval-at-end", action="store_true")
    p.add_argument("--eval-games", type=int, default=12)
    p.add_argument("--eval-depth", type=int, default=2)
    p.add_argument("--lock-memory", action="store_true", help="pin the tables in RAM so paging cannot stall training")
    p.add_argument("--choose", action="store_true",
                   help="train for cool mode: the agent places every tile (implied when resuming a --choose run)")
    p.add_argument("--max-moves", type=int, default=200_000,
                   help="cut choose-mode games at this many moves; a guard only, see above")
    p.add_argument("--explore", type=float, default=0.0,
                   help="choose-mode training: probability per step of placing a random tile instead of the searched one")
    p.add_argument("--starts", default=None,
                   help="choose-mode training: .npy of saved late boards (see ntuple.harvest_starts) to restart games from")
    p.add_argument("--start-frac", type=float, default=0.5,
                   help="share of choose-mode games that begin from a saved board when --starts is given")
    p.add_argument("--eval-prefix", type=int, default=0,
                   help="choose-mode evaluations: kind random spawns for this many moves first (games still "
                        "funnel into one line, so 2 eval games are enough)")
    args = p.parse_args(argv)
    if args.games is None and args.hours is None:
        args.hours = 1.0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state_path, log_path, eval_path = out / "state.json", out / "log.csv", out / "eval.csv"

    boundaries = [int(x) for x in args.stages.split(",") if x.strip()]
    tc_stages = 0 if args.no_tc else args.tc_stages
    if args.resume:
        net = nt.NTupleNet.load(args.resume, boundaries=boundaries or None, tc_stages=tc_stages)
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
    else:
        net = nt.NTupleNet(patterns=nt.PATTERN_SETS[args.patterns], tc=not args.no_tc, boundaries=boundaries,
                           tc_stages=tc_stages)
        state = {}
    choose = bool(args.choose or state.get("choose", False))
    starts = nt.load_starts(args.starts) if args.starts else []
    if starts:
        print(f"{len(starts)} saved late boards; {args.start_frac:.0%} of games restart from one", flush=True)
    locked = bool(args.lock_memory) and net.lock_memory()
    if args.lock_memory and not locked:
        print("warning: could not lock the tables in memory", flush=True)
    games = int(state.get("games", 0))
    moves = int(state.get("moves", 0))
    elapsed_before = float(state.get("elapsed_s", 0.0))
    resumed = bool(args.resume)
    log_f, log = _writer(log_path, LOG_FIELDS, append=resumed and log_path.exists())
    eval_f, evlog = _writer(eval_path, EVAL_FIELDS, append=resumed and eval_path.exists())

    # monotonic pauses while the machine sleeps, so a nap does not consume the budget
    t0 = time.monotonic()
    elapsed = lambda: time.monotonic() - t0 + elapsed_before  # noqa: E731
    last_save = last_eval = time.monotonic()
    games_this_run = 0

    def save() -> None:
        net.save(out / "latest.bin")
        net.save(out / "weights.bin", weights_only=True)  # small deployment copy for the UI
        state_path.write_text(json.dumps({"games": games, "moves": moves, "elapsed_s": elapsed(),
                                          "patterns": net.patterns, "tc": net.tc,
                                          "boundaries": net.boundaries, "threads": args.threads,
                                          "chunk": args.chunk, "alpha": args.alpha, "alpha_plain": args.alpha_plain,
                                          "tc_stages": net.tc_stages, "locked": locked,
                                          "choose": choose, "max_moves": args.max_moves,
                                          "eval_prefix": args.eval_prefix, "explore": args.explore,
                                          "starts": args.starts, "start_frac": args.start_frac if starts else 0.0}))

    def evaluate() -> None:
        t = time.monotonic()
        seed = args.seed + 7919 + games
        if choose:
            played = net.play_choose(args.eval_games, depth=args.eval_depth, topk=4, threads=args.threads,
                                     seed=seed, max_moves=args.max_moves, prefix=args.eval_prefix)
        else:
            played = net.play(args.eval_games, depth=args.eval_depth, threads=args.threads, seed=seed)
        stats = nt.summarize(played)
        row = {"games": games, "elapsed_s": round(elapsed(), 1), "mode": "choose" if choose else "random",
               "depth": args.eval_depth, "eval_games": args.eval_games,
               **{k: stats[k] for k in EVAL_FIELDS if k in stats}}
        evlog.writerow(row)
        eval_f.flush()
        print(f"[eval depth {args.eval_depth}{' choose' if choose else ''}] {args.eval_games} games in {time.monotonic() - t:.0f}s | "
              f"mean {stats['mean_score']:.0f}, max {stats['max_score']} | tile {stats['max_tile']} | "
              f"2048 {stats['rate_2048']:.0%}, 4096 {stats['rate_4096']:.0%}, 8192 {stats['rate_8192']:.0%}, "
              f"16384 {stats['rate_16384']:.0%}, 32768 {stats['rate_32768']:.0%}", flush=True)

    try:
        while True:
            if args.games is not None and games_this_run >= args.games:
                break
            if args.hours is not None and time.monotonic() - t0 >= args.hours * 3600:
                break
            n = args.chunk if args.games is None else min(args.chunk, args.games - games_this_run)
            t = time.monotonic()
            if choose:
                st = net.train_choose(n, threads=args.threads, alpha=args.alpha, seed=args.seed + games,
                                      max_moves=args.max_moves, alpha_plain=args.alpha_plain, explore=args.explore,
                                      starts=starts, start_frac=args.start_frac if starts else 0.0)
            else:
                st = net.train(n, threads=args.threads, alpha=args.alpha, seed=args.seed + games,
                               alpha_plain=args.alpha_plain)
            dt = time.monotonic() - t
            chunk_moves = int(st["moves"].sum())
            games += n
            games_this_run += n
            moves += chunk_moves
            s = nt.summarize(st)
            row = {"games": games, "chunk_moves": chunk_moves, "moves": moves, "elapsed_s": round(elapsed(), 1),
                   "moves_per_s": round(chunk_moves / dt), "mean_score": round(s["mean_score"]),
                   "median_score": round(s["median_score"]), "max_score": s["max_score"],
                   "mean_length": round(s["mean_length"]), "max_tile": s["max_tile"],
                   **{k: round(s[k], 4) for k in ("rate_2048", "rate_4096", "rate_8192", "rate_16384", "rate_32768")}}
            log.writerow(row)
            log_f.flush()
            print(f"games {games} | {moves / 1e6:.1f}M moves | {elapsed() / 60:.0f} min | {row['moves_per_s'] / 1e6:.2f}M moves/s "
                  f"| mean {row['mean_score']} | max {row['max_score']} | tile {row['max_tile']} "
                  f"| 2048 {s['rate_2048']:.0%} 4096 {s['rate_4096']:.0%} 8192 {s['rate_8192']:.0%} "
                  f"16384 {s['rate_16384']:.0%} 32768 {s['rate_32768']:.0%}", flush=True)
            if args.save_every_min and time.monotonic() - last_save >= args.save_every_min * 60:
                save()
                last_save = time.monotonic()
            if args.eval_every_min and time.monotonic() - last_eval >= args.eval_every_min * 60:
                evaluate()
                last_eval = time.monotonic()
    except KeyboardInterrupt:
        print("interrupted; saving", flush=True)

    save()
    if args.eval_at_end:
        evaluate()
    log_f.close()
    eval_f.close()
    print(json.dumps({"games": games, "moves": moves, "elapsed_s": round(elapsed(), 1), "weights": str(out / "latest.bin")}), flush=True)
    net.close()


if __name__ == "__main__":
    main()
