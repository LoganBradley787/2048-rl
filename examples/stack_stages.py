"""Join networks trained one stage at a time into one weights file for play.

    uv run python examples/stack_stages.py --weights checkpoints/ntuple/weights.bin checkpoints/ntuple_stage2/weights.bin --tiles 16384 --out checkpoints/ntuple_2stage/weights.bin

The first network plays until the board holds the first of --tiles, the second
from there, and so on. Boundaries follow `ntuple.stage_of`.
"""

from __future__ import annotations

import argparse

from game2048 import ntuple as nt


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", nargs="+", required=True, help="one file per stage, earliest first")
    p.add_argument("--tiles", required=True, help="comma-separated boundaries, one fewer than the files")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    nets = [nt.NTupleNet.load(w, keep_tc=False) for w in args.weights]
    both = nt.NTupleNet.stack(nets, [int(x) for x in args.tiles.split(",") if x.strip()])
    both.save(args.out, weights_only=True)
    print(f"wrote {args.out}: {len(nets)} stages, boundaries {both.boundaries}")


if __name__ == "__main__":
    main()
