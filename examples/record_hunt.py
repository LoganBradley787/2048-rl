"""Play whole games, one after another, until one beats the single-game record.

    uv run python examples/record_hunt.py --out checkpoints/hunt/s777.csv --seed 777 --weights ... [play options]

The play options are evaluate_games.py's, and so are the files: every game is a row of the CSV as
it finishes, the settings go to the .json beside it (a CSV played with other settings is refused),
and `evaluate_games.py --summary` reads them. Stop it at any time; started again with the same
command it plays the games an interrupted run left unfinished, then goes on from the next number.

Every game is recorded move by move as it is played; the ones scoring --trace-min-score or more are
kept (game_<n>.npz in --trace-dir) once they pass a check against the rules. A game scoring
--record-score or more gets RECORD_seed<seed>_game<n>.json beside the CSV (game number, seed,
settings, engine fingerprint, git commit, trace and its sha256), then --notify-cmd is run, then the
game is played again and compared with its trace board by board (the result goes into the RECORD
file and a second notification). examples/verify_game.py does the same check on its own later.

--notify-cmd is a command line; the message and a title are added as its last two arguments, or
put where it says {msg} and {title}, e.g. --notify-cmd 'notes/notify.sh {msg} {title} urgent'.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_games as eg  # noqa: E402
import verify_game  # noqa: E402

from game2048 import ntuple as nt  # noqa: E402

RECORD = 1_704_908                                     # the best single game known when this was written
ROOT = Path(__file__).resolve().parents[1]


def git_state() -> dict:
    def git(*a: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, timeout=30,
                                  check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": None if status is None else bool(status)}


def file_hash(path: Path, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def notify_argv(cmd: str, msg: str, title: str) -> list[str]:
    parts = shlex.split(cmd)
    if any("{msg}" in x or "{title}" in x for x in parts):
        return [x.replace("{msg}", msg).replace("{title}", title) for x in parts]
    return [*parts, msg, title]


def game_order(done, limit: int):
    """The games an earlier run left unplayed below its last one, then the next numbers on (to `limit`, or
    without end when it is 0)."""
    top = max(done) + 1 if done else 0
    yield from (g for g in range(top if not limit else min(top, limit)) if g not in done)
    g = top
    while not limit or g < limit:
        yield g
        g += 1


def write_json(path: Path, doc: dict) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="CSV of finished games; its settings go to the .json beside it")
    p.add_argument("--games", type=int, default=0, help="stop once games 0 .. n-1 are played; 0 = never stop")
    eg.add_play_args(p)
    eg.add_trace_args(p, trace_min=1_500_000, report_min=1_000_000)
    p.add_argument("--record-score", type=int, default=RECORD,
                   help=f"a game scoring this or more is a record: RECORD_ file and notification (default {RECORD:,})")
    p.add_argument("--notify-cmd", default=None,
                   help="run with the message and a title on every record (and --notify-min-score game); default none")
    p.add_argument("--notify-min-score", type=int, default=0,
                   help="also notify for games scoring this or more that are not records; 0 = records only")
    p.add_argument("--summary-every", type=int, default=256, help="print throughput and a summary every n games")
    args = p.parse_args(argv)

    out, path = Path(args.out), nt.resolve_weights(args.weights)
    out.parent.mkdir(parents=True, exist_ok=True)
    args.trace_dir = args.trace_dir or str(out.with_name(out.stem + "_traces"))
    caps = eg.check_args(p, args)
    if args.trace_min_score > args.record_score:
        p.error("--trace-min-score above --record-score would leave a record without its moves")
    settings = eg.play_settings(args, path, caps)
    settings["games"] = args.games or None
    eg.write_settings(out, settings)
    meta = out.with_suffix(".json").resolve()

    done = eg.read_games(out)
    order = game_order(set(done), args.games)
    first = next(order, None)
    if first is None:
        print(f"{out}: all {args.games} games played", flush=True)
        eg.show(eg.summarize([out]))
        return 0
    holes = sum(1 for g in range(max(done) + 1 if done else 0) if g not in done)
    print(f"{path}: depth {args.depth}, seed {args.seed}: {len(done):,} games done, {holes} unfinished below "
          f"the last; playing from game {first} on {args.threads} threads "
          f"({'up to game ' + str(args.games - 1) if args.games else 'without end'})", flush=True)

    net = eg.load_net(args, path, caps)
    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    provenance = {**git_state(), "engine_md5": settings["engine_md5"], "engine_lib_md5": file_hash(nt.LIB, "md5"),
                  "host": socket.gethostname()}
    print(f"git {provenance['git_commit']}, engine source {provenance['engine_md5']}, library "
          f"{provenance['engine_lib_md5']}, traces of {args.trace_min_score:,}+ in {trace_dir}", flush=True)
    if provenance["git_dirty"]:
        print("note: the git checkout has uncommitted changes; a record's commit will not hold its exact code", flush=True)

    new = not out.exists()
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=eg.FIELDS)
    if new:
        w.writeheader()
        f.flush()
    lock, stop, errors, notes = threading.Lock(), threading.Event(), [], []
    tally = {"n": len(done), "sum": sum(int(r["score"]) for r in done.values()), "best": (-1, -1),
             "65536": sum(int(r["max_tile"]) >= 65536 for r in done.values()),
             "1M": sum(int(r["score"]) >= 1_000_000 for r in done.values()),
             "traced": sum(int(r["score"]) >= args.trace_min_score for r in done.values()), "session": 0}
    for g, r in done.items():
        tally["best"] = max(tally["best"], (int(r["score"]), -g))
    t0 = time.time()

    def notify(msg: str, title: str = "2048 record hunt") -> None:
        print(f"notify: {msg}", flush=True)
        if not args.notify_cmd:
            return
        def run():
            try:
                subprocess.run(notify_argv(args.notify_cmd, msg, title), timeout=900)
            except (OSError, subprocess.SubprocessError) as e:
                print(f"notify failed: {e}", flush=True)
        th = threading.Thread(target=run, daemon=True)
        th.start()
        notes.append(th)

    def summary() -> str:
        hours = (time.time() - t0) / 3600
        best, g = tally["best"][0], -tally["best"][1]
        return (f"{tally['n']:,} games | {tally['session'] / hours if hours else 0:,.0f} games/hour this run | "
                f"mean {tally['sum'] / max(1, tally['n']):,.0f} | best {best:,} (game {g}) | "
                f"65536 {100 * tally['65536'] / max(1, tally['n']):.1f}% | >= 1M: {tally['1M']} | "
                f">= {args.trace_min_score:,}: {tally['traced']}")

    def record(g: int, row: dict) -> None:
        name = out.parent / f"RECORD_seed{args.seed}_game{g}.json"
        trace = Path(row["trace"]).resolve() if row["trace"] else None
        doc = {"what": f"game {g} of seed {args.seed} scored {row['score']:,} (record threshold {args.record_score:,})",
               "game": g, "seed": args.seed, "score": row["score"], "max_tile": row["max_tile"], "moves": row["moves"],
               "seconds": row["seconds"], "found": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
               "trace": str(trace) if trace else None, "trace_sha256": file_hash(trace) if trace else None,
               "csv": str(out.resolve()), "settings_file": str(meta), "settings": settings, **provenance,
               "replay_check": "pending",
               "verify": f"uv run python examples/verify_game.py {name.resolve()}"}
        write_json(name, doc)
        notify(f"RECORD? game {g} seed {args.seed}: {row['score']:,} points, max tile {row['max_tile']}"
               + ("; trace kept, replay check running" if trace else "; NO TRACE (refused), see the log"),
               "2048 RECORD")
        if not trace:
            return
        try:
            ok, report = verify_game.verify(settings, g, trace, net=net)
        except Exception as e:                         # noqa: BLE001 - the record stands; the check is redone by hand
            ok, report = False, f"FAIL game {g}: the replay check raised {e!r}"
        print(report, flush=True)
        doc.update(replay_check="OK" if ok else "FAIL", replay_report=report)
        write_json(name, doc)
        notify(f"record game {g}: replay check {'OK' if ok else 'FAILED'}", "2048 RECORD")

    def worker() -> None:
        try:
            while not stop.is_set():
                with lock:
                    g = next(pending, None)
                if g is None:
                    return
                t = time.time()
                row = eg.play_game(net, g, args, trace_dir, args.trace_min_score)
                row["seconds"] = round(time.time() - t, 1)
                with lock:                             # one row per finished game, on disk at once
                    w.writerow({"game": g, **{k: row[k] for k in eg.FIELDS if k != "game"}})
                    f.flush()
                    os.fsync(f.fileno())
                    tally["n"] += 1
                    tally["session"] += 1
                    tally["sum"] += row["score"]
                    tally["best"] = max(tally["best"], (row["score"], -g))
                    tally["65536"] += row["max_tile"] >= 65536
                    tally["1M"] += row["score"] >= 1_000_000
                    tally["traced"] += row["score"] >= args.trace_min_score
                    if args.report_min_score and row["score"] >= args.report_min_score:
                        print(f"game {g}: score {row['score']:,}, max tile {row['max_tile']}, {row['moves']} moves"
                              + (f", trace {row['trace']}" if row["trace"] else ""), flush=True)
                    if tally["session"] % max(1, args.summary_every) == 0:
                        print(summary(), flush=True)
                if row["score"] >= args.record_score:
                    record(g, row)
                elif args.notify_min_score and row["score"] >= args.notify_min_score:
                    notify(f"game {g} seed {args.seed}: {row['score']:,} points, max tile {row['max_tile']}")
        except BaseException as e:                     # noqa: BLE001 - reported, and the hunt stops
            errors.append(e)
            stop.set()

    pending = (g for it in ([first], order) for g in it)
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, args.threads))]
    for th in threads:
        th.start()
    try:
        while any(th.is_alive() for th in threads) and not errors:
            time.sleep(1.0)
    except KeyboardInterrupt:
        stop.set()
        print("stopped; the games in play are lost and will be played again on the next start", flush=True)
    finally:
        with lock:
            errs = list(errors)                        # the games still in play end here; restarting plays them
            f.close()
    print(summary(), flush=True)
    eg.report_tables(args)
    if errs:
        print(f"ERROR: {errs[0]!r}", flush=True)
        notify(f"record hunt {out.name} stopped on an error: {errs[0]!r}"[:300])
    for th in notes:                                   # let the last notifications go out
        th.join(timeout=300)
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
