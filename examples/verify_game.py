"""Check a kept game: play it again from scratch and compare it with its trace, board by board.

    uv run python examples/verify_game.py checkpoints/hunt/hunt.json 123456
    uv run python examples/verify_game.py checkpoints/hunt/RECORD_seed777_game123456.json

The first form takes the settings .json written beside a games CSV (by evaluate_games.py or
record_hunt.py) and a game number; the trace defaults to <trace_dir>/game_<n>.npz. The second takes
a RECORD_ file, which holds both. The trace is also checked against the rules of 2048 alone (every
board follows from the one before by its move and one new tile; the points add up). Prints OK or
FAIL; the exit status is 0 or 1. Run it with the same weights and tables (checked by fingerprint)
and, to be sure of the engine, at the git commit the game was played with.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_games as eg  # noqa: E402
import rules2048  # noqa: E402

from game2048 import ntuple as nt  # noqa: E402


def load_trace(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def replay(net: nt.NTupleNet, settings: dict, game: int, trace: dict) -> list[str]:
    """Game `game` played again with `net` (set up from `settings`), compared with `trace`.
    Returns what differs (empty: the same game)."""
    bad = []
    for k, want in (("game", game), ("seed", settings["seed"])):
        if k in trace and int(trace[k]) != int(want):
            bad.append(f"the trace is of {k} {int(trace[k])}, not {want}")
    boards, dirs = trace["boards"], trace["dirs"]
    s = net.play_trace(game, depth=settings["depth"], cutoff=settings["cutoff"], seed=settings["seed"],
                       max_boards=max(len(boards) + 1, eg.TRACE_CAP))
    for k in ("score", "max_tile", "moves"):
        if k in trace and int(trace[k]) != int(s[k]):
            bad.append(f"{k}: trace {int(trace[k])}, played again {s[k]}")
    if len(s["boards"]) != len(boards):
        bad.append(f"{len(boards)} boards in the trace, {len(s['boards'])} played again")
    n = min(len(boards), len(s["boards"]))
    differ = np.flatnonzero((s["boards"][:n] != boards[:n]).reshape(n, -1).any(axis=1) | (s["dirs"][:n] != dirs[:n]))
    if len(differ):
        bad.append(f"{len(differ)} boards or moves differ, the first at move {int(differ[0])}")
    return bad


def verify(settings: dict, game: int, trace_path: Path, net: nt.NTupleNet | None = None) -> tuple[bool, str]:
    """(ok, report). Without `net`, one is set up from `settings` (tables load once per process)."""
    trace = load_trace(trace_path)
    problems = []
    rules = eg.check_rules(trace["boards"], trace["dirs"], int(trace["score"]))
    if rules:
        problems.append(f"against the rules: {rules}")
    own_rules = rules2048.check_trace(trace["boards"], trace["dirs"], int(trace["score"]))   # shares no engine code
    if own_rules:
        problems.append(f"against the rules (independent check): {own_rules}")
    p = argparse.ArgumentParser()
    eg.add_play_args(p)
    args = p.parse_args(eg.args_from_settings(settings))
    path = Path(args.weights)
    missing = [x for x in [path, *(args.block_dir or []), *(args.endgame_dir or [])] if not Path(x).exists()]
    if missing:
        return False, f"FAIL game {game}: cannot play it again, missing {', '.join(map(str, missing))}"
    if eg.weights_md5(path) != settings["weights_md5"]:
        problems.append(f"{path} is not the weights the game was played with")
    engine = hashlib.md5(nt.SRC.read_bytes()).hexdigest()
    if engine != settings["engine_md5"]:
        problems.append(f"engine source {engine} is not the one the game was played with ({settings['engine_md5']})")
    tables = settings.get("block_tables")
    if tables and tables != " | ".join((Path(d) / "block.txt").read_text().strip() for d in args.block_dir):
        problems.append("the block tables are not the ones the game was played with")
    own = net is None
    if own:
        net = eg.load_net(args, path, eg.check_args(p, args))
    t = time.time()
    try:
        problems += replay(net, settings, game, trace)
    finally:
        if own:
            net.close()
    head = (f"game {game} of seed {settings['seed']}: score {int(trace['score'])}, max tile {int(trace['max_tile'])}, "
            f"{len(trace['boards']) - 1} moves ({trace_path}; played again in {time.time() - t:.0f} s)")
    if problems:
        return False, "FAIL " + head + "\n  " + "\n  ".join(problems)
    return True, "OK " + head + ": every board and move the same, and every move legal (two independent rule checks)"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("settings", help="settings .json beside a games CSV, or a RECORD_<game>.json")
    p.add_argument("game", type=int, nargs="?", default=None, help="game number (taken from a RECORD_ file)")
    p.add_argument("--trace", default=None, help="the trace .npz (default: from the settings' trace_dir)")
    args = p.parse_args(argv)
    doc = json.loads(Path(args.settings).read_text())
    settings = doc.get("settings", doc)
    game = args.game if args.game is not None else doc.get("game")
    if game is None:
        p.error("which game? (a game number, or a RECORD_ file)")
    if args.trace:
        trace = Path(args.trace)
    elif doc.get("trace"):
        trace = Path(doc["trace"])
    else:
        if not settings.get("trace_dir"):
            p.error("these settings kept no traces; give --trace")
        trace = Path(settings["trace_dir"]) / f"game_{game}.npz"
    if not trace.is_absolute() and not trace.exists() and settings.get("cwd"):
        trace = Path(settings["cwd"]) / trace
    if not trace.exists():
        print(f"FAIL game {game}: no trace at {trace}", flush=True)
        return 1
    ok, report = verify(settings, int(game), trace)
    print(report, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
