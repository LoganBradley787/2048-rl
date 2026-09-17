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
    with a beam search over the deterministic game: 32 lines, 12 steps ahead, and
    4 steps of each plan played before searching again (a quarter of the cost for
    the same score)."""
    from .ntuple import NTupleAgent, resolve_cool_weights

    return NTupleAgent(weights=resolve_cool_weights(), beam_width=32, beam_depth=12, beam_stride=4)


AGENTS: dict[str, Callable[[], Agent]] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "nn": _nn_agent,
    "ntuple": _ntuple_agent,
    "ntuple-cool": _ntuple_cool_agent,
}


def make_agent(name: str) -> Agent:
    try:
        return AGENTS[name]()
    except KeyError:
        raise KeyError(f"unknown agent {name!r}; known: {sorted(AGENTS)}") from None
