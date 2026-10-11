"""Collect the boards where one stage hands over to the next: play games with
trained weights and keep, from each, the first board that holds --tile.

    uv run python examples/collect_starts.py --weights checkpoints/ntuple/weights.bin --tile 16384 --boards 200000 --out checkpoints/starts_16384.npy

The next stage trains from them: `python -m game2048.ntuple_train --starts ...`.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from game2048 import ntuple as nt


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=None, help="defaults to checkpoints/ntuple/weights.bin or latest.bin")
    p.add_argument("--tile", type=int, default=16384)
    p.add_argument("--boards", type=int, default=200_000, help="stop once this many are collected")
    p.add_argument("--depth", type=int, default=0, help="search depth of the games played (0 = no search, as in training)")
    p.add_argument("--chunk", type=int, default=20_000, help="games per native call")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    net = nt.NTupleNet.load(nt.resolve_weights(args.weights), keep_tc=False)
    kept, points, games, t0 = [], [], 0, time.time()
    while sum(len(k) for k in kept) < args.boards:
        grids, scores = net.collect_starts(args.chunk, tile=args.tile, depth=args.depth, threads=args.threads,
                                           seed=args.seed + games)
        games += args.chunk
        kept.append(grids)
        points.append(scores)
        n = sum(len(k) for k in kept)
        print(f"{games} games | {n} boards ({n / games:.0%} of games reach {args.tile}) | {time.time() - t0:.0f}s", flush=True)
        if n == 0 and games >= 5 * args.chunk:
            raise SystemExit(f"no game reached {args.tile}; these weights are not ready for the next stage")
    grids = np.concatenate(kept)[: args.boards]
    nt.save_starts(args.out, grids)
    print(f"wrote {len(grids)} boards to {args.out}; mean score at the hand-over {np.concatenate(points).mean():.0f}")
    net.close()


if __name__ == "__main__":
    main()
