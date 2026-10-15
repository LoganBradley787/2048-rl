"""Build the endgame tables for `evaluate_games.py --endgame K --endgame-dir DIR`.

    uv run python examples/build_endgame.py --kmin 5 --out tables/k5 --threads 16

Compiles game2048/native/endgame_build.c (OpenMP if the compiler has it) and runs it.
Defaults: caps on the tables with 5 and 6 locked tiles only (drops 0,1,2,3,3,4,4,5,5,5,5),
every larger table exact. Sizes at 2 bytes a position: 6 locked and up 3.4 GB; 5 locked and
up another 33 GB. The output is written in place on disk; memory use stays small.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "game2048" / "native" / "endgame_build.c"


def compile_builder(out: Path) -> None:
    cc = os.environ.get("CC", "cc")
    # -ffp-contract=off: the same arithmetic on every CPU, so a table built anywhere is the same file
    base = [cc, "-O3", "-ffp-contract=off", "-DEGB_MAIN", "-o", str(out), str(SRC), "-lm"]
    try:
        subprocess.run(base[:1] + ["-fopenmp"] + base[1:], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("no OpenMP: building on one thread", file=sys.stderr)
        subprocess.run(base, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kmin", type=int, default=6, help="smallest table: this many locked tiles (5 to 15)")
    p.add_argument("--drops", default="0,1,2,3,3,4,4,5,5,5,5", help="caps: free cell j holds at most 2^(n - d_j)")
    p.add_argument("--kcap", type=int, default=6, help="caps apply to tables with at most this many locked tiles")
    p.add_argument("--path", default="snake",
                   help="formation: snake, block (32768 16384 8192 on top, 4096 2048 1024 under them), or 16 cells")
    p.add_argument("--lift", type=int, default=0,
                   help="sub-goal tables: the last locked tile is this many sizes below the usual one (1: a 1024 on "
                        "the 5th cell, tiles in play up to it), for the phases where a gap in the ladder is being filled")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "endgame_build"
        compile_builder(exe)
        path = {"snake": None, "block": "0,1,2,6,5,4,8,9,10,11,7,3,15,14,13,12"}.get(args.path, args.path)
        cmd = [str(exe), str(args.kmin), args.drops, str(out), str(args.threads), str(args.kcap)]
        if path or args.lift:
            cmd += [path or "-"] + ([str(args.lift)] if args.lift else [])
        subprocess.run(cmd, check=True)
    print(f"tables in {out}: " + ", ".join(f"{f.name} {f.stat().st_size / 1e9:.2f} GB" for f in sorted(out.glob("k*.u16"))
                                       if f.stat().st_size > 1e8))


if __name__ == "__main__":
    main()
