"""Board evaluators for the best/evil spawn modes.

`best_available()` returns the strongest evaluator on this machine: the
trained n-tuple tables when they exist, else `core.heuristic_value`.
"""

from __future__ import annotations

from typing import Callable

from .core import heuristic_value

Evaluator = Callable[[list[list[int]]], float]


def ntuple_evaluator(weights=None) -> Evaluator:
    from . import ntuple as nt  # compiles the C library on first use

    net = nt.NTupleNet.load(nt.resolve_weights(weights))

    def value(board: list[list[int]]) -> float:
        return net.value(nt.tiles_to_bits(board))

    value.net = net  # keep the tables alive with the closure
    return value


def best_available() -> Evaluator:
    try:
        return ntuple_evaluator()
    except (FileNotFoundError, ImportError, OSError, ValueError):
        return heuristic_value
