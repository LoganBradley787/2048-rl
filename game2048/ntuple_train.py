"""Train the n-tuple network with TD(0) self-play in native code.

    uv run python -m game2048.ntuple_train --hours 8 --threads 12

Writes to --out: latest.bin (weights + learning state), weights.bin (weights
only, what the UI loads), state.json (counters), log.csv (one row per chunk
of training games) and eval.csv (expectimax evaluations). Resume with
--resume <out>/latest.bin. `kill <pid>` (or Ctrl-C) stops a run after the chunk
in progress and saves first.

--choose trains for cool mode (the player places every tile): self-play uses
the choose search and evaluations play the choose game. Keep --max-moves far
above the natural game length; games cut at the cap leave their late boards
ungrounded, values inflate and the policy collapses.

A later stage is trained as a network of its own, from the boards where an
earlier one hands over (examples/collect_starts.py writes them):

    uv run python -m game2048.ntuple_train --starts checkpoints/starts_16384.npy --base checkpoints/ntuple/weights.bin --stage-tile 16384 --out checkpoints/ntuple_stage2 ...

Every game then begins from one of those boards, so the log's scores count from
there; the evaluations play whole games with --base below the tile and this
network from it on. examples/stack_stages.py joins the two for play.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import time
from pathlib import Path

import numpy as np

from . import ntuple as nt

LOG_FIELDS = [
    "games", "chunk_moves", "moves", "elapsed_s", "moves_per_s", "mean_score", "median_score",
    "max_score", "mean_length", "max_tile", "rate_2048", "rate_4096", "rate_8192", "rate_16384",
    "rate_32768", "rate_65536",
]
EVAL_FIELDS = [
    "games", "elapsed_s", "mode", "depth", "eval_games", "mean_score", "median_score", "max_score",
    "mean_length", "max_tile", "rate_2048", "rate_4096", "rate_8192", "rate_16384", "rate_32768",
    "rate_65536",
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
    p.add_argument("--total-moves", type=float, default=None,
                   help="stop once the network has trained on this many moves in all, earlier runs included "
                        "(e.g. 9e11); the way to give a run that is stopped and resumed a fixed budget")
    p.add_argument("--chunk", type=int, default=5000, help="games per native call and per log row")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    p.add_argument("--alpha", type=float, default=1.0, help="step for temporal-coherence stages")
    p.add_argument("--alpha-plain", type=float, default=0.025,
                   help="total step for plain-TD stages (only used with --no-tc or --tc-stages)")
    p.add_argument("--tc-stages", type=int, default=None,
                   help="leading stages that keep temporal-coherence state (default: all, or 0 with --no-tc)")
    p.add_argument("--no-tc", action="store_true",
                   help="plain TD instead of temporal coherence (with --resume: drop the TC state, a third of the memory)")
    p.add_argument("--expect-target", action="store_true",
                   help="learn from the average over every tile that could appear instead of the one that did "
                        "(single-stage, random-spawn runs; 10 to 15 times the cost per move); pass it again on resume")
    p.add_argument("--stages", default="", help="comma-separated boundaries for multi-stage tables, on the tiles a board holds: "
                        "16384 = a 16384 or larger, 24576 = a 16384 and an 8192, or larger")
    p.add_argument("--patterns", default="4x6", choices=sorted(nt.PATTERN_SETS), help="pattern set for a new network")
    p.add_argument("--v-init", type=float, default=0.0,
                   help="value a new network gives every board (optimistic initialisation); 320000 for 8x6")
    p.add_argument("--out", default="checkpoints/ntuple")
    p.add_argument("--resume", default=None, help="weights file to continue from")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every-min", type=float, default=10.0, help="0 = only at the end")
    p.add_argument("--keep-previous", action="store_true",
                   help="keep the checkpoint before the latest as previous.bin, in case the machine dies mid-save")
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
                   help=".npy of saved boards to begin games from: every game (see examples/collect_starts.py), "
                        "or in choose mode a --start-frac share of them (see ntuple.harvest_starts)")
    p.add_argument("--start-frac", type=float, default=0.5,
                   help="share of choose-mode games that begin from a saved board when --starts is given")
    p.add_argument("--base", default=None,
                   help="with --starts: weights of the earlier stage, so evaluations play whole games (implied on resume)")
    p.add_argument("--stage-tile", type=int, default=16384,
                   help="with --base: the tile from which this network takes over")
    p.add_argument("--eval-prefix", type=int, default=0,
                   help="choose-mode evaluations: kind random spawns for this many moves first (games still "
                        "funnel into one line, so 2 eval games are enough)")
    args = p.parse_args(argv)
    if args.games is None and args.hours is None and args.total_moves is None:
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
                           tc_stages=tc_stages, v_init=args.v_init)
        state = {"v_init": args.v_init}
    choose = bool(args.choose or state.get("choose", False))
    if not choose:                                     # a stage run keeps its boards and its base when resumed
        args.starts = args.starts or (state.get("starts") if args.resume else None)
        args.base = args.base or (state.get("base") if args.resume else None)
        if args.resume and state.get("base"):
            args.stage_tile = int(state.get("stage_tile", args.stage_tile))
    starts = nt.load_starts(args.starts) if args.starts and choose else []
    if starts:
        print(f"{len(starts)} saved late boards; {args.start_frac:.0%} of games restart from one", flush=True)
    packed = nt.pack_starts(np.load(args.starts)) if args.starts and not choose else None
    if packed:
        print(f"{packed[1]} saved boards; every game begins from one and its score counts from there", flush=True)
    base = nt.NTupleNet.load(args.base, keep_tc=False) if args.base and not choose else None
    if args.expect_target:
        if choose or net.boundaries:
            p.error("--expect-target is for single-stage runs on the normal game")
        net.expect_target = True
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
        # Each file is written beside its place and renamed over it, so a run that is
        # killed mid-save leaves the last complete checkpoint, never half of a new one.
        net.save(out / "latest.bin.tmp")
        if args.keep_previous and (out / "latest.bin").exists():
            os.replace(out / "latest.bin", out / "previous.bin")
        os.replace(out / "latest.bin.tmp", out / "latest.bin")
        net.save(out / "weights.bin.tmp", weights_only=True)  # small deployment copy for the UI
        os.replace(out / "weights.bin.tmp", out / "weights.bin")
        state_tmp = out / "state.json.tmp"
        state_tmp.write_text(json.dumps({"games": games, "moves": moves, "elapsed_s": elapsed(),
                                          "patterns": net.patterns, "tc": net.tc,
                                          "boundaries": net.boundaries, "threads": args.threads,
                                          "chunk": args.chunk, "alpha": args.alpha, "alpha_plain": args.alpha_plain,
                                          "tc_stages": net.tc_stages, "locked": locked,
                                          "v_init": state.get("v_init", 0.0),
                                          "choose": choose, "max_moves": args.max_moves,
                                          "eval_prefix": args.eval_prefix, "explore": args.explore,
                                          "starts": args.starts, "start_frac": args.start_frac if starts else 0.0,
                                          "base": args.base, "stage_tile": args.stage_tile if base else 0,
                                          "expect_target": bool(args.expect_target)}))
        os.replace(state_tmp, state_path)

    def evaluate() -> None:
        t = time.monotonic()
        seed = args.seed + 7919 + games
        if choose:
            played = net.play_choose(args.eval_games, depth=args.eval_depth, topk=4, threads=args.threads,
                                     seed=seed, max_moves=args.max_moves, prefix=args.eval_prefix)
        elif base:                                     # whole games: the earlier stage, then this one
            both = nt.NTupleNet.stack([base, net], [args.stage_tile])
            played = both.play(args.eval_games, depth=args.eval_depth, threads=args.threads, seed=seed)
            both.close()
        else:
            played = net.play(args.eval_games, depth=args.eval_depth, threads=args.threads, seed=seed)
        stats = nt.summarize(played)
        row = {"games": games, "elapsed_s": round(elapsed(), 1), "mode": "choose" if choose else "random",
               "depth": args.eval_depth, "eval_games": args.eval_games,
               **{k: stats[k] for k in EVAL_FIELDS if k in stats and k != "games"}}   # games = training games so far
        evlog.writerow(row)
        eval_f.flush()
        print(f"[eval depth {args.eval_depth}{' choose' if choose else ''}] {args.eval_games} games in {time.monotonic() - t:.0f}s | "
              f"mean {stats['mean_score']:.0f}, max {stats['max_score']} | tile {stats['max_tile']} | "
              f"2048 {nt.percent(stats['rate_2048'])}, 4096 {nt.percent(stats['rate_4096'])}, 8192 {nt.percent(stats['rate_8192'])}, "
              f"16384 {nt.percent(stats['rate_16384'])}, 32768 {nt.percent(stats['rate_32768'], 1)}", flush=True)

    # A detached run (nohup ... &) starts with SIGINT ignored, so Python never installs
    # its Ctrl-C handler; take both signals here so `kill` stops the run with a save.
    def _stop(signum, frame):
        raise KeyboardInterrupt

    old_handlers = {sig: signal.signal(sig, _stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while True:
            if args.games is not None and games_this_run >= args.games:
                break
            if args.hours is not None and time.monotonic() - t0 >= args.hours * 3600:
                break
            if args.total_moves is not None and moves >= args.total_moves:
                break
            n = args.chunk if args.games is None else min(args.chunk, args.games - games_this_run)
            t = time.monotonic()
            if choose:
                st = net.train_choose(n, threads=args.threads, alpha=args.alpha, seed=args.seed + games,
                                      max_moves=args.max_moves, alpha_plain=args.alpha_plain, explore=args.explore,
                                      starts=starts, start_frac=args.start_frac if starts else 0.0)
            else:
                st = net.train(n, threads=args.threads, alpha=args.alpha, seed=args.seed + games,
                               alpha_plain=args.alpha_plain, starts=packed)
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
                   **{k: round(s[k], 4) for k in ("rate_2048", "rate_4096", "rate_8192", "rate_16384", "rate_32768",
                                                  "rate_65536")}}
            log.writerow(row)
            log_f.flush()
            print(f"games {games} | {moves / 1e6:.1f}M moves | {elapsed() / 60:.0f} min | {row['moves_per_s'] / 1e6:.2f}M moves/s "
                  f"| mean {row['mean_score']} | max {row['max_score']} | tile {row['max_tile']} "
                  f"| 2048 {nt.percent(s['rate_2048'])} 4096 {nt.percent(s['rate_4096'])} 8192 {nt.percent(s['rate_8192'])} "
                  f"16384 {nt.percent(s['rate_16384'])} 32768 {nt.percent(s['rate_32768'], 1)}", flush=True)
            if args.save_every_min and time.monotonic() - last_save >= args.save_every_min * 60:
                save()
                last_save = time.monotonic()
            if args.eval_every_min and time.monotonic() - last_eval >= args.eval_every_min * 60:
                evaluate()
                last_eval = time.monotonic()
    except KeyboardInterrupt:
        print("interrupted; saving", flush=True)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)

    save()
    if args.eval_at_end:
        evaluate()
    log_f.close()
    eval_f.close()
    print(json.dumps({"games": games, "moves": moves, "elapsed_s": round(elapsed(), 1), "weights": str(out / "latest.bin")}), flush=True)
    net.close()


if __name__ == "__main__":
    main()
