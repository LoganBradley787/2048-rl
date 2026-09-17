"""Evaluate a trained checkpoint over many games.

    uv run python examples/evaluate.py --games 500
    uv run python examples/evaluate.py --games 500 --symmetric   # average the 8 board symmetries
"""

from __future__ import annotations

import argparse
import time

from game2048 import nn as g2nn
from game2048.train import evaluate_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(g2nn.DEFAULT_CHECKPOINT))
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--json", help="also write the stats (plus checkpoint meta) to this file")
    args = parser.parse_args()

    device = g2nn.pick_device(args.device)
    model, scale, meta = g2nn.load_checkpoint(args.checkpoint, device)
    print(f"checkpoint: {args.checkpoint} (iteration {meta.get('iteration')}, "
          f"{meta.get('env_steps', 0) / 1e6:.1f}M env steps, {meta.get('games')} training games)")
    t = time.time()
    stats = evaluate_policy(model, args.games, scale, device, seed=args.seed, symmetric=args.symmetric)
    print(f"{args.games} games in {time.time() - t:.1f}s, symmetric={args.symmetric}\n")
    print(f"mean score   {stats['mean_score']:>9.0f}")
    print(f"median score {stats['median_score']:>9.0f}")
    print(f"max score    {stats['max_score']:>9d}")
    print(f"mean length  {stats['mean_length']:>9.0f} moves\n")
    print("max tile reached:")
    for tile, count in sorted(stats["tile_counts"].items()):
        print(f"  {tile:>6d}  {count / args.games:6.1%}  {'#' * int(50 * count / args.games)}")
    print(f"\nreached 1024: {stats['rate_1024']:.1%}   2048: {stats['rate_2048']:.1%}   "
          f"4096: {stats['rate_4096']:.1%}   8192: {stats['rate_8192']:.1%}")
    if args.json:
        import json
        from pathlib import Path

        Path(args.json).write_text(json.dumps(
            {**stats, "symmetric": args.symmetric, "device": str(device), "meta": meta}, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
