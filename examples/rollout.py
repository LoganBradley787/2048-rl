"""Run a few episodes with the built-in agents through Env2048.

This is the shape a training loop takes: reset, loop over step() until done.
Swap the agent for a network that maps `obs` (16 log2 exponents) to an action.

    python examples/rollout.py --agent greedy --episodes 20
"""

from __future__ import annotations

import argparse
import statistics

from game2048.agents import make_agent
from game2048.env import Env2048


def run_episode(env: Env2048, agent) -> tuple[int, int, int]:
    env.reset()
    done = False
    steps = 0
    while not done:
        action = agent.act(env.game)
        _, _, done, info = env.step(action)
        steps += 1
    return info["score"], info["max_tile"], steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="greedy")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scores, tiles, lengths = [], [], []
    for ep in range(args.episodes):
        env = Env2048(seed=args.seed + ep)
        score, max_tile, steps = run_episode(env, make_agent(args.agent))
        scores.append(score)
        tiles.append(max_tile)
        lengths.append(steps)
        print(f"episode {ep:3d}  score {score:6d}  max tile {max_tile:5d}  steps {steps:4d}")

    print(
        f"\n{args.agent}: mean score {statistics.mean(scores):.0f}, "
        f"best score {max(scores)}, best tile {max(tiles)}, "
        f"mean length {statistics.mean(lengths):.0f}"
    )


if __name__ == "__main__":
    main()
