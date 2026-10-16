"""Build the endgame tables for `evaluate_games.py --endgame K --endgame-dir DIR`.

    uv run python examples/build_endgame.py --kmin 5 --out tables/k5 --threads 16

Compiles game2048/native/endgame_build.c (OpenMP if the compiler has it) and runs it.
Defaults: caps on the tables with 5 and 6 locked tiles only (drops 0,1,2,3,3,4,4,5,5,5,5),
every larger table exact. Sizes at 2 bytes a position: 6 locked and up 3.4 GB; 5 locked and
up another 33 GB. The output is written in place on disk; memory use stays small.

Block-tier tables (game2048/native/block_build.c; play with nt.load_block_tables and net.block_tier):

    uv run python examples/build_endgame.py --block b6 --anchors 9,10 --out tables/b6 --threads 4
    uv run python examples/build_endgame.py --block b6 --sizes      # the size of every table, no build

writes <out>/<shape>_a<a>.u16 per anchor and <out>/block.txt; the sizes are printed before building.
With --bytes 1 a table is <shape>_a<a>.mv8, half the size: each position's best move and a 6-bit value (the
same moves as the u16 table; block_build.c), built with three layers of u16 values in memory (0.45 GB at b5 a 10)
and no u16 file:

    uv run python examples/build_endgame.py --block b5 --anchors 10 --bytes 1 --out tables/b5 --threads 12
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "game2048" / "native" / "endgame_build.c"
BLOCK_SRC = SRC.with_name("block_build.c")


def compile_builder(out: Path, src: Path = SRC) -> None:
    cc = os.environ.get("CC", "cc")
    # -ffp-contract=off: the same arithmetic on every CPU, so a table built anywhere is the same file
    base = [cc, "-O3", "-ffp-contract=off", "-DEGB_MAIN", "-o", str(out), str(src), "-lm"]
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
    p.add_argument("--wobble", action="store_true",
                   help="the locked tiles after the last full row may slide along their row (snake only)")
    p.add_argument("--block", choices=["b6", "b5"],
                   help="block-tier tables of this shape instead (b6: 2x3 block in the corner; b5: 3 + 2 cells)")
    p.add_argument("--anchors", default="", help="with --block: the anchors (tile ranks) to build, e.g. 9,10")
    p.add_argument("--sizes", action="store_true", help="with --block: print every table's size and stop")
    p.add_argument("--bytes", type=int, choices=[1, 2], default=2,
                   help="with --block: 2 = u16 values (default); 1 = mv8, a best move and a 6-bit value a position")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    p.add_argument("--out")
    args = p.parse_args()
    if args.block:
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "block_build"
            compile_builder(exe, BLOCK_SRC)
            if args.sizes:
                subprocess.run([str(exe), "sizes"], check=True)
                return
            if not args.out or not args.anchors:
                p.error("--block needs --anchors and --out")
            Path(args.out).mkdir(parents=True, exist_ok=True)
            subprocess.run([str(exe), args.block, args.anchors, args.out, str(args.threads), str(args.bytes)],
                           check=True)
        print(f"block tables in {args.out}: " + (Path(args.out) / "block.txt").read_text().strip())
        return
    if not args.out:
        p.error("--out is required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "endgame_build"
        compile_builder(exe)
        path = {"snake": None, "block": "0,1,2,6,5,4,8,9,10,11,7,3,15,14,13,12"}.get(args.path, args.path)
        cmd = [str(exe), str(args.kmin), args.drops, str(out), str(args.threads), str(args.kcap)]
        if path or args.lift or args.wobble:
            cmd += [path or "-"] + ([str(args.lift)] if args.lift or args.wobble else [])
        if args.wobble:
            cmd += ["1"]
        subprocess.run(cmd, check=True)
    print(f"tables in {out}: " + ", ".join(f"{f.name} {f.stat().st_size / 1e9:.2f} GB" for f in sorted(out.glob("k*.u16"))
                                       if f.stat().st_size > 1e8))


if __name__ == "__main__":
    main()
