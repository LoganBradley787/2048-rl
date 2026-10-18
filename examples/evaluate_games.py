"""A large evaluation that can be stopped, resumed and split across machines.

Every game is written to a CSV as it finishes. Run the same command again and
it plays only the games that are missing. A game depends on the seed and its
number alone, so shards run on different machines add up to one evaluation:

    uv run python examples/evaluate_games.py --weights checkpoints/ntuple/weights.bin --depth 3 --games 1000 --shard 0/2 --out checkpoints/eval/d3-a.csv
    uv run python examples/evaluate_games.py --weights checkpoints/ntuple/weights.bin --depth 3 --games 1000 --shard 1/2 --out checkpoints/eval/d3-b.csv
    uv run python examples/evaluate_games.py --summary checkpoints/eval/d3-a.csv checkpoints/eval/d3-b.csv

The settings are kept in a .json beside each CSV; a resumed run or a summary
refuses files whose settings differ.

--trace-dir keeps games move by move (--trace-min-score: only the games that
score at least that much); examples/verify_game.py replays a kept game and
checks it board by board.
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
            "endgame_bonus", "heuristic", "heuristic_prune", "heuristic_depth", "heuristic_from", "block_tables", "block_from", "deep"]
TRACE_CAP = 200_000                                    # boards kept during play; a longer game is played again


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


def add_play_args(p: argparse.ArgumentParser) -> None:
    """The options that decide how the games are played (shared with record_hunt.py and verify_game.py)."""
    p.add_argument("--weights", default=None, help="defaults to checkpoints/ntuple/weights.bin or latest.bin")
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
    p.add_argument("--heuristic-prune", nargs="?", const="small", choices=["small", "big"], default=None,
                   help="macroxue's pruning beside his tables; 'big': his big build's rules (prune only from a healthy "
                        "root, search again unpruned when the best move scores below 0)")
    p.add_argument("--heuristic-depth", type=int, default=0,
                   help="with --heuristic: search the late boards this deep (0 = --depth; --depth-caps still apply)")
    p.add_argument("--heuristic-from", type=int, default=0,
                   help="with --heuristic: it plays boards whose tile set is at least this (0 = 49152, a 32768 and "
                        "a 16384; 32768: from the first 32768)")
    p.add_argument("--deep", default="",
                   help="D:T: search at least D deep once the board holds a tile T (e.g. 2:8192 with --depth 1: depth 1 "
                        "until the first 8192, then 2); several tiles as D:T1,T2 (all of them on the board)")
    p.add_argument("--block-dir", action="append", default=None,
                   help="block-tier tables (examples/build_endgame.py --block) play the boards no ladder table holds; "
                        "repeat it for several shapes")
    p.add_argument("--block-from", type=int, default=0,
                   help="with --block-dir: only boards whose tile set reaches this go to the block tables, e.g. 49152 "
                        "(a 32768 and a 16384, or larger: the boards --heuristic plays); 0 = every board")


def add_trace_args(p: argparse.ArgumentParser, trace_min: int = 0, report_min: int = 0) -> None:
    p.add_argument("--trace-dir", default=None,
                   help="keep games' histories here, one game_<n>.npz each: `boards` (the board before each "
                        "move, then the final one, as exponents) and `dirs` (0 up, 1 right, 2 down, 3 left, -1 at the end)")
    p.add_argument("--trace-min-score", type=int, default=trace_min,
                   help=f"with --trace-dir: keep only the games scoring this or more (default {trace_min:,}). Every "
                        "game is recorded as it is played, at no extra cost; only these are written")
    p.add_argument("--report-min-score", type=int, default=report_min,
                   help=f"print a line at once for every game scoring this or more (default {report_min:,}; 0 = off)")


def check_args(p: argparse.ArgumentParser, args) -> list[int]:
    """Option errors; returns the depth caps."""
    if args.heuristic == "off" and (args.heuristic_prune or args.heuristic_depth or args.heuristic_from):
        p.error("--heuristic-prune, --heuristic-depth and --heuristic-from need --heuristic")
    if args.block_from and not args.block_dir:
        p.error("--block-from needs --block-dir")
    if getattr(args, "trace_min_score", 0) and not getattr(args, "trace_dir", None):
        p.error("--trace-min-score needs --trace-dir")
    return [int(x) for x in args.depth_caps.split(",") if x.strip()]


def play_settings(args, path: Path, caps: list[int]) -> dict:
    """What the games depend on (and what is needed to play one again), for the .json beside the CSV."""
    trace_dir = getattr(args, "trace_dir", None)
    return {"weights_md5": weights_md5(path), "depth": args.depth, "seed": args.seed, "cutoff": args.cutoff,
            "downgrade": args.downgrade, "depth_caps": caps, "endgame": args.endgame or None,
            "endgame_bits": args.endgame_bits if args.endgame else None,
            "endgame_dir": args.endgame_dir, "endgame_bonus": args.endgame_bonus or None,
            # the engine's source: a run resumed under another engine would mix two players
            "engine_md5": hashlib.md5(nt.SRC.read_bytes()).hexdigest(),
            "endgame_caps": args.endgame_dir and " | ".join((Path(d) / "caps.txt").read_text().strip()
                                                            for d in args.endgame_dir), "weights": str(path),
            "games": getattr(args, "games", None),
            # off = absent, so files written before these options existed still match
            "heuristic": None if args.heuristic == "off" else args.heuristic,
            "heuristic_prune": {"small": True, "big": "big"}.get(args.heuristic_prune), "heuristic_depth": args.heuristic_depth or None,
            "heuristic_from": args.heuristic_from or None,
            "block_dir": args.block_dir,
            "block_tables": args.block_dir and " | ".join((Path(d) / "block.txt").read_text().strip()
                                                          for d in args.block_dir),
            "block_from": args.block_from or None,
            "deep": args.deep or None,
            # not compared between files (they do not change the games); kept so that a game can be played again
            "tt_bits": args.tt_bits or None, "cwd": os.getcwd(), "trace_dir": trace_dir,
            "trace_min_score": getattr(args, "trace_min_score", 0) if trace_dir else None}


def write_settings(out: Path, settings: dict) -> None:
    """The .json beside the CSV; refuses a CSV whose games were played with other settings."""
    meta = out.with_suffix(".json")
    if meta.exists():
        old = json.loads(meta.read_text())
        diff = {k: (old.get(k), settings[k]) for k in SETTINGS if old.get(k) != settings[k]}
        if diff:
            raise SystemExit(f"{out} holds games played with other settings {diff}; use another --out")
        for k in ("trace_dir", "trace_min_score"):     # the games in the file were traced under these
            if k in old and old[k] != settings[k]:
                raise SystemExit(f"{out} was played with {k} {old[k]!r}, not {settings[k]!r}; use the same or another --out")
    meta.write_text(json.dumps(settings, indent=1))


def args_from_settings(s: dict) -> list[str]:
    """The play options of a settings .json as command-line arguments (relative paths are taken from the
    directory the games were played in)."""
    def at(x: str) -> str:
        q = Path(x)
        return str(q if q.is_absolute() or not s.get("cwd") else Path(s["cwd"]) / q)
    argv = ["--weights", at(s["weights"]), "--depth", str(s["depth"]), "--seed", str(s["seed"]),
            "--cutoff", repr(float(s["cutoff"])), "--downgrade", str(s["downgrade"]),
            "--depth-caps", ",".join(str(c) for c in s.get("depth_caps") or []), "--tt-bits", str(s.get("tt_bits") or 0)]
    if s.get("endgame"):
        argv += ["--endgame", str(s["endgame"]), "--endgame-bits", str(s["endgame_bits"])]
        argv += [a for d in s.get("endgame_dir") or [] for a in ("--endgame-dir", at(d))]
        if s.get("endgame_bonus"):
            argv += ["--endgame-bonus", repr(float(s["endgame_bonus"]))]
    if s.get("heuristic"):
        argv += ["--heuristic", s["heuristic"]]
        if s.get("heuristic_prune"):
            argv += ["--heuristic-prune", "big" if s["heuristic_prune"] == "big" else "small"]
        if s.get("heuristic_depth"):
            argv += ["--heuristic-depth", str(s["heuristic_depth"])]
        if s.get("heuristic_from"):
            argv += ["--heuristic-from", str(s["heuristic_from"])]
    argv += [a for d in s.get("block_dir") or [] for a in ("--block-dir", at(d))]
    if s.get("block_from"):
        argv += ["--block-from", str(s["block_from"])]
    if s.get("deep"):
        argv += ["--deep", s["deep"]]
    return argv


def load_net(args, path: Path, caps: list[int]) -> nt.NTupleNet:
    """The network with every play option set (tables load once per process)."""
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
                raise SystemExit(f"--endgame {args.endgame} is below the lowest table in {d} ({low})")
            print(f"endgame tables built in advance from {low} locked up ({d}, {'snake' if snake else words[4]})", flush=True)
        net.endgame = args.endgame
        net.endgame_bonus = args.endgame_bonus
    for d in args.block_dir or []:
        print(f"block-tier tables, anchors {nt.load_block_tables(d)} ({d})", flush=True)
    net.block_tier, net.block_tier_from = bool(args.block_dir), args.block_from
    if args.heuristic != "off":
        net.heuristic_orientation = args.heuristic
        net.heuristic_late, net.heuristic_depth = True, args.heuristic_depth
        net.heuristic_from = args.heuristic_from
        net.heuristic_prune = {"small": True, "big": "big"}.get(args.heuristic_prune, False)
    if args.deep:
        d, tiles = args.deep.split(":")
        net.deep = (int(d), [int(t) for t in tiles.split(",")])
    net.depth_caps = caps
    if args.tt_bits:
        net.tt_bits = args.tt_bits
    return net


def check_rules(boards: np.ndarray, dirs: np.ndarray, score: int) -> str | None:
    """A trace checked against the rules of 2048 alone: it starts with two 2s or 4s; each board is the one
    before after its move, plus one 2 or 4 in an empty cell; the points add up to `score`; the last board
    has no move. Returns what is wrong, or None."""
    if len(boards) == 0 or len(boards) != len(dirs) or dirs[-1] != -1 or (dirs[:-1] < 0).any() or (dirs[:-1] > 3).any():
        return "the moves are not a finished game"
    first = boards[0].reshape(16)
    if np.count_nonzero(first) != 2 or not set(first[first > 0].tolist()) <= {1, 2}:
        return "the first board is not two 2s or 4s"
    b = [sum(int(v) << (nt.CELL_BITS * i) for i, v in enumerate(row)) for row in boards.reshape(-1, 16).tolist()]
    total = 0
    for t in range(len(b) - 1):
        after, r = nt.move(b[t], int(dirs[t]))
        if after == b[t]:
            return f"move {t} does not change the board"
        total += r
        d = b[t + 1] - after
        c = (d.bit_length() - 1) // nt.CELL_BITS if d > 0 else 0
        if d not in (1 << (nt.CELL_BITS * c), 2 << (nt.CELL_BITS * c)) or nt.cell(after, c) != 0:
            return f"board {t + 1} is not board {t} after move {int(dirs[t])} plus one new 2 or 4"
    if any(nt.move(b[-1], m)[0] != b[-1] for m in range(4)):
        return "the last board still has a move"
    if total != score:
        return f"the moves score {total}, not {score}"
    return None


def save_trace(path: Path, s: dict, game: int, seed: int) -> None:
    """Written whole or not at all."""
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, boards=s["boards"], dirs=s["dirs"], score=s["score"], max_tile=s["max_tile"],
                            moves=s["moves"], game=game, seed=seed)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def play_game(net: nt.NTupleNet, g: int, args, trace_dir: Path | None = None, trace_min: int = 0) -> dict:
    """Game g of args.seed: score, max_tile, moves, and `trace`, the file its history went to (or None).
    With a trace_dir every game is recorded as it is played (play_trace plays exactly as play does, at the
    same cost); one scoring trace_min or more is kept once it passes check_rules. A game longer than the
    buffer is played again in full and kept only if the second play is the same game."""
    if trace_dir is None:
        s = net.play(1, depth=args.depth, cutoff=args.cutoff, threads=1, seed=args.seed, first=g)
        return {"score": int(s["scores"][0]), "max_tile": int(s["max_tiles"][0]), "moves": int(s["moves"][0]),
                "trace": None}
    s = net.play_trace(g, depth=args.depth, cutoff=args.cutoff, seed=args.seed, max_boards=TRACE_CAP)
    row = {"score": s["score"], "max_tile": s["max_tile"], "moves": s["moves"], "trace": None}
    if s["score"] < trace_min:
        return row
    if len(s["boards"]) != s["moves"] + 1:             # the buffer was too short: the same game again, all of it
        again = net.play_trace(g, depth=args.depth, cutoff=args.cutoff, seed=args.seed, max_boards=s["moves"] + 1)
        n = len(s["boards"])
        if (again["score"], again["max_tile"], again["moves"]) != (s["score"], s["max_tile"], s["moves"]) or \
                not np.array_equal(again["boards"][:n], s["boards"]) or not np.array_equal(again["dirs"][:n], s["dirs"]):
            print(f"TRACE REFUSED game {g}: played again it was not the same game "
                  f"(score {s['score']}, then {again['score']})", flush=True)
            return row
        s = again
    bad = check_rules(s["boards"], s["dirs"], s["score"])
    if bad:
        print(f"TRACE REFUSED game {g}: {bad}", flush=True)
        return row
    path = trace_dir / f"game_{g}.npz"
    save_trace(path, s, g, args.seed)
    row["trace"] = str(path)
    return row


def report_tables(args) -> None:
    if args.block_dir:
        print(f"block tier: {nt.block_tier_moves():,} moves chosen", flush=True)
    if args.endgame:                                   # a table that ran out of room makes play slow and timing-dependent
        L = nt.lib()
        print("endgame tables: positions kept " + ", ".join(f"{k} locked {L.nt_endgame_used(k):,}" for k in range(5, 8))
              + (" | RAN OUT OF ROOM: values from then on may be floors; rerun with a larger --endgame-bits"
                 if L.nt_endgame_full() else " | never full"), flush=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", nargs="+", default=None, help="CSV files to merge and report; plays nothing")
    p.add_argument("--out", default=None, help="CSV of finished games; its settings go to the .json beside it")
    p.add_argument("--games", type=int, default=1000, help="size of the whole evaluation: games 0 .. n-1 of the seed")
    p.add_argument("--shard", default="0/1", help="i/n: play the games whose number is i modulo n")
    add_play_args(p)
    add_trace_args(p)
    args = p.parse_args(argv)

    if args.summary:
        show(summarize([Path(x) for x in args.summary]))
        return
    if not args.out:
        p.error("--out is required")
    out, path = Path(args.out), nt.resolve_weights(args.weights)
    out.parent.mkdir(parents=True, exist_ok=True)
    caps = check_args(p, args)
    write_settings(out, play_settings(args, path, caps))

    i, n = (int(x) for x in args.shard.split("/"))
    done = read_games(out)
    todo = [g for g in range(args.games) if g % n == i and g not in done]
    print(f"{path}: depth {args.depth}, shard {i}/{n}: {len(done)} games done, {len(todo)} to play "
          f"on {args.threads} threads", flush=True)
    if not todo:
        show(summarize([out]))
        return

    net = load_net(args, path, caps)
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
        row = play_game(net, g, args, trace_dir, args.trace_min_score)
        row.pop("trace")
        with lock:                                     # one row per finished game, on disk at once
            w.writerow({"game": g, **row, "seconds": round(time.time() - t, 1)})
            f.flush()
            os.fsync(f.fileno())
            count[0] += 1
            if args.report_min_score and row["score"] >= args.report_min_score:
                print(f"game {g}: score {row['score']}, max tile {row['max_tile']}, {row['moves']} moves", flush=True)
            if count[0] % max(1, args.threads) == 0 or count[0] == len(todo):
                print(f"{count[0]}/{len(todo)} games in {(time.time() - t0) / 60:.1f} min", flush=True)

    try:
        with ThreadPoolExecutor(args.threads) as pool:
            list(pool.map(play, todo))
    finally:
        f.close()
    show(summarize([out]))
    report_tables(args)
    net.close()


if __name__ == "__main__":
    main()
