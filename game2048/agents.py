"""Pluggable policies. Anything with `act(game) -> Direction` is an agent.

To watch a trained network play in the browser, add an entry to `AGENTS`
whose factory returns an object exposing `act(game)`. Use `Env2048.observe`
-style encoding on `game.board` inside `act` and mask to `game.legal_moves()`.
"""

from __future__ import annotations

import random
from typing import Callable, Protocol

from .core import Direction, Game


class Agent(Protocol):
    def act(self, game: Game) -> Direction: ...


class RandomAgent:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def act(self, game: Game) -> Direction:
        return self._rng.choice(game.legal_moves())


class GreedyAgent:
    """Highest immediate reward; ties broken by fewest tiles remaining."""

    def act(self, game: Game) -> Direction:
        best: tuple[int, int, Direction] | None = None
        for d in game.legal_moves():
            trial = game.clone()
            _, reward = trial.move(d)
            tiles = sum(1 for row in trial.board for v in row if v)
            key = (reward, -tiles, d)
            if best is None or key[:2] > best[:2]:
                best = key
        if best is None:
            raise ValueError("no legal moves")
        return best[2]


def _nn_agent() -> Agent:
    from .nn import NNAgent  # torch is only needed for this agent

    return NNAgent()


def _ntuple_agent() -> Agent:
    from .ntuple import NTupleAgent  # compiles the C library on first use

    return NTupleAgent()


def _ntuple_cool_agent() -> Agent:
    """n-tuple tables trained for cool mode (the agent places every tile), played
    with a beam search over the deterministic game. The weights are the best
    scoring snapshot when one has been promoted (checkpoints/ntuple_best), and the
    beam settings come from the best.json beside them; otherwise 32 lines, 12 steps
    ahead, 4 steps of each plan before searching again, snake bonus 1.0."""
    from .ntuple import NTupleAgent, cool_search_settings, resolve_cool_weights

    weights = resolve_cool_weights()
    cfg = cool_search_settings(weights)
    return NTupleAgent(weights=weights, beam_width=cfg["width"], beam_depth=cfg["depth"],
                       beam_stride=cfg["stride"], beam_snake=cfg["snake"])


def _snake_agent() -> Agent:
    """No training at all: beam search over the choose game scoring boards by their
    merges plus the snake-order heuristic (an untouched network contributes 0).
    64 lines, 16 steps, snake weight 2: 1.61M with a 65536 on the canonical game."""
    from .ntuple import NTupleAgent, NTupleNet

    return NTupleAgent(net=NTupleNet(patterns=[[0, 1, 2, 3]], tc=False), beam_width=64, beam_depth=16,
                       beam_stride=4, beam_snake=2.0)


AGENTS: dict[str, Callable[[], Agent]] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "nn": _nn_agent,
    "ntuple": _ntuple_agent,
    "ntuple-cool": _ntuple_cool_agent,
    "snake": _snake_agent,
}


def make_agent(name: str) -> Agent:
    try:
        return AGENTS[name]()
    except KeyError:
        raise KeyError(f"unknown agent {name!r}; known: {sorted(AGENTS)}") from None
