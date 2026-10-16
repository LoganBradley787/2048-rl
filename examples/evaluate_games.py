"""A large evaluation that can be stopped, resumed and split across machines.

Every game is written to a CSV as it finishes. Run the same command again and
it plays only the games that are missing. A game depends on the seed and its
number alone, so shards run on different machines add up to one evaluation:

    uv run python examples/evaluate_games.py --weights checkpoints/ntuple/weights.bin --depth 3 --games 1000 --shard 0/2 --out checkpoints/eval/d3-a.csv
    uv run python examples/evaluate_games.py --weights checkpoints/ntuple/weights.bin --depth 3 --games 1000 --shard 1/2 --out checkpoints/eval/d3-b.csv
    uv run python examples/evaluate_games.py --summary checkpoints/eval/d3-a.csv checkpoints/eval/d3-b.csv

The settings are kept in a .json beside each CSV; a resumed run or a summary
refuses files whose settings differ.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from game2048 import ntuple as nt

FIELDS = ["game", "score", "max_tile", "moves", "seconds"]
SETTINGS = ["weights_md5", "depth", "seed", "cutoff", "downgrade", "depth_caps", "endgame", "endgame_caps", "engine_md5",
            "endgame_bonus", "heuristic", "heuristic_prune", "heuristic_depth", "block_tables"]


def weights_md5(path: Path) -> str:
    """Fingerprint of a weights file: its size and its first and last megabyte."""
    size, h = path.stat().st_size, hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        f.seek(max(0, size - (1 << 20)))
        h.update(f.read(1 << 20))
    return f"{size}-{h.hexdigest()[:16]}"


def read_games(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {int(r["game"]): r for r in csv.DictReader(f)}


def summarize(paths: list[Path]) -> dict:
    settings, games = None, {}
    for p in paths:
        s = json.loads(p.with_suffix(".json").read_text())
        s = {k: s.get(k) for k in SETTINGS}
        if settings is not None and s != settings:
            raise SystemExit(f"{p} was played with different settings: {s} against {settings}")
        settings = s
        games.update(read_games(p))
    if not games:
        raise SystemExit("no games yet")
    scores = np.array([int(g["score"]) for g in games.values()], dtype=np.int64)
    stats = nt.summarize({"scores": scores, "max_tiles": np.array([int(g["max_tile"]) for g in games.values()]),
                          "moves": np.array([int(g["moves"]) for g in games.values()])})
    stats["ci95"] = float(1.96 * scores.std(ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else float("nan")
    stats["game_hours"] = sum(float(g["seconds"]) for g in games.values()) / 3600
    stats["settings"] = settings
    return stats


def show(stats: dict) -> None:
    print(f"{stats['games']} games | mean {stats['mean_score']:.0f} +- {stats['ci95']:.0f} (95%) | "
          f"median {stats['median_score']:.0f} | max {stats['max_score']} | {stats['mean_length']:.0f} moves | "
          f"8192 {nt.percent(stats['rate_8192'], 1)}, 16384 {nt.percent(stats['rate_16384'], 1)}, "
          f"32768 {nt.percent(stats['rate_32768'], 1)}, 65536 {nt.percent(stats['rate_65536'], 1)} | "
          f"{stats['game_hours']:.1f} thread-hours", flush=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", nargs="+", default=None, help="CSV files to merge and report; plays nothing")
    p.add_argument("--weights", default=None, help="defaults to checkpoints/ntuple/weights.bin or latest.bin")
    p.add_argument("--out", default=None, help="CSV of finished games; its settings go to the .json beside it")
    p.add_argument("--games", type=int, default=1000, help="size of the whole evaluation: games 0 .. n-1 of the seed")
    p.add_argument("--shard", default="0/1", help="i/n: play the games whose number is i modulo n")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--threads", type=int, default=os.cpu_count() or 4, help="games played at once")
    p.add_argument("--cutoff", type=float, default=0.0)
    p.add_argument("--downgrade", type=int, default=0, help="tile (e.g. 32768) from which the search uses the downgraded board")
    p.add_argument("--tt-bits", type=int, default=0, help="transposition table of 2**n entries per game in play; 0 = default")
    p.add_argument("--depth-caps", default="", help="deepest search by number of empty cells, e.g. 5,5,5,5,4,4,4,4,3")
    p.add_argument("--endgame", type=int, default=0,
                   help="exact endgame tables choose the move once this many tiles (32768 down) lie on a snake; 0 = off")
    p.add_argument("--endgame-dir", action="append", default=None,
                   help="tables built in advance (examples/build_endgame.py) instead of solving on demand; repeat it for "
                        "several formations (play takes the best)")
    p.add_argument("--endgame-bonus", type=float, default=0.0,
                   help="points times the table's chance for a move into a table position (tables built in advance)")
    p.add_argument("--endgame-bits", type=int, default=29,
                   help="memo size for the 5-tile table as a power of two; 6 tiles get the same up to 31, 7 tiles two fewer up to 27. "
                        "Under 31: 7 bytes a slot (29 is 3.8 GB); 31 and over: 4 bytes (33 is 32 GB)")
    p.add_argument("--heuristic", choices=["off", *nt.HEURISTIC_ORIENTATIONS], default="off",
                   help="late boards (a 32768 and a 16384, or a 65536 and up) searched with macroxue's hand-written "
                        "heuristic, the board turned per move (turn), as it stands (own) or once per game (sticky)")
    p.add_argument("--heuristic-prune", action="store_true",
                   help="with --heuristic: macroxue's pruning and loss value (his settings beside his tables)")
    p.add_argument("--heuristic-depth", type=int, default=0,
                   help="with --heuristic: search the late boards this deep (0 = --depth; --depth-caps still apply)")
    p.add_argument("--block-dir", action="append", default=None,
                   help="block-tier tables (examples/build_endgame.py --block) play the boards no ladder table holds; "
                        "repeat it for several shapes")
    p.add_argument("--trace-dir", default=None,
                   help="keep every game's history here, one game_<n>.npz each: `boards` (the board before each "
                        "move, then the final one, as exponents) and `dirs` (0 up, 1 right, 2 down, 3 left, -1 at the end)")
    args = p.parse_args(argv)

    if args.summary:
        show(summarize([Path(x) for x in args.summary]))
        return
    if not args.out:
        p.error("--out is required")
    out, path = Path(args.out), nt.resolve_weights(args.weights)
    out.parent.mkdir(parents=True, exist_ok=True)
    caps = [int(x) for x in args.depth_caps.split(",") if x.strip()]
    if args.heuristic == "off" and (args.heuristic_prune or args.heuristic_depth):
        p.error("--heuristic-prune and --heuristic-depth need --heuristic")
    settings = {"weights_md5": weights_md5(path), "depth": args.depth, "seed": args.seed, "cutoff": args.cutoff,
                "downgrade": args.downgrade, "depth_caps": caps, "endgame": args.endgame or None,
                "endgame_bits": args.endgame_bits if args.endgame else None,
                "endgame_dir": args.endgame_dir, "endgame_bonus": args.endgame_bonus or None,
                # the engine's source: a run resumed under another engine would mix two players
                "engine_md5": hashlib.md5(nt.SRC.read_bytes()).hexdigest(),
                "endgame_caps": args.endgame_dir and " | ".join((Path(d) / "caps.txt").read_text().strip()
                                                                for d in args.endgame_dir), "weights": str(path), "games": args.games,
                # off = absent, so files written before these options existed still match
                "heuristic": None if args.heuristic == "off" else args.heuristic,
                "heuristic_prune": args.heuristic_prune or None, "heuristic_depth": args.heuristic_depth or None,
                "block_dir": args.block_dir,
                "block_tables": args.block_dir and " | ".join((Path(d) / "block.txt").read_text().strip()
                                                              for d in args.block_dir)}
    meta = out.with_suffix(".json")
    if meta.exists():
        old = json.loads(meta.read_text())
        diff = {k: (old.get(k), settings[k]) for k in SETTINGS if old.get(k) != settings[k]}
        if diff:
            raise SystemExit(f"{out} holds games played with other settings {diff}; use another --out")
    meta.write_text(json.dumps(settings, indent=1))

    i, n = (int(x) for x in args.shard.split("/"))
    done = read_games(out)
    todo = [g for g in range(args.games) if g % n == i and g not in done]
    print(f"{path}: depth {args.depth}, shard {i}/{n}: {len(done)} games done, {len(todo)} to play "
          f"on {args.threads} threads", flush=True)
    if not todo:
        show(summarize([out]))
        return

    net = nt.NTupleNet.load(path, keep_tc=False)
    net.downgrade = args.downgrade
    if args.endgame:
        nt.lib().nt_endgame_bits(args.endgame_bits)
        for d in args.endgame_dir or []:
            words = (Path(d) / "caps.txt").read_text().split()
            snake = (len(words) < 5 or [int(x) for x in words[4].split(",")] == nt.SNAKE_PATH) and \
                (len(words) < 6 or words[5] == "0")
            low = nt.load_endgame(d)
            if snake and args.endgame < low:           # below the snake's files its tables would be solved during play
                p.error(f"--endgame {args.endgame} is below the lowest table in {d} ({low})")
            print(f"endgame tables built in advance from {low} locked up ({d}, {'snake' if snake else words[4]})", flush=True)
        net.endgame = args.endgame
        net.endgame_bonus = args.endgame_bonus
    for d in args.block_dir or []:
        print(f"block-tier tables, anchors {nt.load_block_tables(d)} ({d})", flush=True)
    net.block_tier = bool(args.block_dir)
    if args.heuristic != "off":
        net.heuristic_orientation = args.heuristic
        net.heuristic_late, net.heuristic_prune, net.heuristic_depth = True, args.heuristic_prune, args.heuristic_depth
    net.depth_caps = caps
    if args.tt_bits:
        net.tt_bits = args.tt_bits
    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    if trace_dir:
        trace_dir.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    lock, t0, count = threading.Lock(), time.time(), [0]

    def play(g: int) -> None:
        t = time.time()
        if trace_dir:                                  # the same game, with its history kept
            s = net.play_trace(g, depth=args.depth, cutoff=args.cutoff, seed=args.seed)
            np.savez_compressed(trace_dir / f"game_{g}.npz", boards=s["boards"], dirs=s["dirs"], score=s["score"])
            row = {"score": s["score"], "max_tile": s["max_tile"], "moves": s["moves"]}
        else:
            s = net.play(1, depth=args.depth, cutoff=args.cutoff, threads=1, seed=args.seed, first=g)
            row = {"score": int(s["scores"][0]), "max_tile": int(s["max_tiles"][0]), "moves": int(s["moves"][0])}
        with lock:                                     # one row per finished game, on disk at once
            w.writerow({"game": g, **row, "seconds": round(time.time() - t, 1)})
            f.flush()
            os.fsync(f.fileno())
            count[0] += 1
            if count[0] % max(1, args.threads) == 0 or count[0] == len(todo):
                print(f"{count[0]}/{len(todo)} games in {(time.time() - t0) / 60:.1f} min", flush=True)

    try:
        with ThreadPoolExecutor(args.threads) as pool:
            list(pool.map(play, todo))
    finally:
        f.close()
    show(summarize([out]))
    if args.block_dir:
        print(f"block tier: {nt.block_tier_moves():,} moves chosen", flush=True)
    if args.endgame:                                   # a table that ran out of room makes play slow and timing-dependent
        L = nt.lib()
        print("endgame tables: positions kept " + ", ".join(f"{k} locked {L.nt_endgame_used(k):,}" for k in range(5, 8))
              + (" | RAN OUT OF ROOM: values from then on may be floors; rerun with a larger --endgame-bits"
                 if L.nt_endgame_full() else " | never full"), flush=True)
    net.close()


if __name__ == "__main__":
    main()
