"""Pure-Python 2048 rules. No dependencies, so it can be used directly in a
training loop without going through HTTP.

Action indices are fixed: UP=0, RIGHT=1, DOWN=2, LEFT=3.

Spawn modes decide where the new tile goes after each move:
  random  the standard game: uniform empty cell, 2 with p=0.9 else 4
  kind    the same, except a tile that would end the game is never chosen
          while some placement keeps a move available
  best    the placement that maximises the player's best reply (reward plus
          an evaluator's value of the resulting afterstate); ties are random
  evil    the placement that minimises it, and ends the game whenever it can
  choose  no automatic spawn: after each move the game waits until the player
          calls place(row, col, 2 or 4) on an empty cell
"""

from __future__ import annotations

import math
import random
from enum import IntEnum
from typing import Callable, NamedTuple

SPAWN_MODES = ("random", "kind", "best", "evil", "choose")


class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class MoveResult(NamedTuple):
    moved: bool
    reward: int


def slide_and_merge_row(row: list[int]) -> tuple[list[int], int]:
    """Slide a row to the left and merge equal neighbours once each.

    Returns the new row (same length) and the points gained (sum of merged
    tiles). The input is not mutated.
    """
    tiles = [v for v in row if v]
    out: list[int] = []
    reward = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            merged = tiles[i] * 2
            out.append(merged)
            reward += merged
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    out.extend([0] * (len(row) - len(out)))
    return out, reward


def heuristic_value(board: list[list[int]]) -> float:
    """Dependency-free stand-in for a learned evaluator: room to move plus
    merges that are ready. Used by best/evil spawns when no model is given."""
    n = len(board)
    empty = sum(1 for row in board for v in row if v == 0)
    pairs = 0
    for i in range(n):
        for j in range(n):
            v = board[i][j]
            if not v:
                continue
            if j + 1 < n and board[i][j + 1] == v:
                pairs += v
            if i + 1 < n and board[i + 1][j] == v:
                pairs += v
    return 128.0 * empty + pairs


class Game:
    """Holds the board and score, applies moves, spawns tiles.

    `board[i][j]` is the tile value at row i, column j (0 = empty).
    """

    SPAWN_FOUR_PROBABILITY = 0.1

    def __init__(self, size: int = 4, seed: int | None = None, spawn_mode: str = "random",
                 evaluator: Callable[[list[list[int]]], float] | None = None) -> None:
        if spawn_mode not in SPAWN_MODES:
            raise ValueError(f"spawn_mode must be one of {SPAWN_MODES}, not {spawn_mode!r}")
        self.size = size
        self.spawn_mode = spawn_mode
        self.evaluator = evaluator or heuristic_value
        self._rng = random.Random(seed)
        self.board: list[list[int]] = []
        self.score = 0
        self.awaiting_tile = False  # choose mode: a move happened and no tile is placed yet
        self.reset()

    # --- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        self.board = [[0] * self.size for _ in range(self.size)]
        self.score = 0
        self.awaiting_tile = False
        self._random_spawn()
        self._random_spawn()

    def clone(self) -> "Game":
        other = Game.__new__(Game)
        other.size = self.size
        other.spawn_mode = self.spawn_mode
        other.evaluator = self.evaluator
        other.board = [row[:] for row in self.board]
        other.score = self.score
        other.awaiting_tile = self.awaiting_tile
        other._rng = random.Random()
        other._rng.setstate(self._rng.getstate())
        return other

    # --- moves -------------------------------------------------------------

    def move(self, direction: Direction | int) -> MoveResult:
        if self.awaiting_tile:
            raise RuntimeError("place a tile before moving again")
        direction = Direction(direction)
        new_board, reward = self._apply(self.board, direction)
        if new_board == self.board:
            return MoveResult(False, 0)
        self.board = new_board
        self.score += reward
        if self.spawn_mode == "choose":
            self.awaiting_tile = True
        else:
            self._spawn()
        return MoveResult(True, reward)

    def place(self, row: int, col: int, value: int) -> None:
        """Choose mode: put the next tile down yourself."""
        if not self.awaiting_tile:
            raise ValueError("the game is not waiting for a tile")
        if value not in (2, 4):
            raise ValueError("a placed tile must be 2 or 4")
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise ValueError("cell is off the board")
        if self.board[row][col]:
            raise ValueError("cell is not empty")
        self.board[row][col] = value
        self.awaiting_tile = False

    def best_placement(self) -> tuple[int, int, int]:
        """Choose mode helper: the placement that maximises the player's best reply
        (the same rule as the best spawn mode). Does not place anything."""
        if not self.awaiting_tile:
            raise ValueError("the game is not waiting for a tile")
        empty = [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i][j] == 0]
        return self._extreme_spawn(empty, best=True)

    def legal_moves(self) -> list[Direction]:
        if self.awaiting_tile:
            return []
        return [d for d in Direction if self._apply(self.board, d)[0] != self.board]

    def is_over(self) -> bool:
        return not self.awaiting_tile and not self.legal_moves()

    def max_tile(self) -> int:
        return max(v for row in self.board for v in row)

    # --- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "board": [row[:] for row in self.board],
            "score": self.score,
            "over": self.is_over(),
            "max_tile": self.max_tile(),
            "legal_moves": [int(d) for d in self.legal_moves()],
            "mode": self.spawn_mode,
            "awaiting_tile": self.awaiting_tile,
        }

    @classmethod
    def from_dict(cls, data: dict, seed: int | None = None,
                  evaluator: Callable[[list[list[int]]], float] | None = None) -> "Game":
        g = cls.__new__(cls)
        g.board = [list(row) for row in data["board"]]
        g.size = len(g.board)
        g.score = int(data.get("score", 0))
        g.spawn_mode = data.get("mode", "random")
        if g.spawn_mode not in SPAWN_MODES:
            raise ValueError(f"unknown spawn mode {g.spawn_mode!r}")
        g.evaluator = evaluator or heuristic_value
        g.awaiting_tile = bool(data.get("awaiting_tile", False))
        g._rng = random.Random(seed)
        return g

    # --- internals ---------------------------------------------------------

    @staticmethod
    def _apply(board: list[list[int]], direction: Direction) -> tuple[list[list[int]], int]:
        """Return (new_board, reward) for `direction` without spawning."""
        # Normalise so that every direction becomes a left-slide of rows.
        if direction in (Direction.UP, Direction.DOWN):
            rows = [list(col) for col in zip(*board)]
        else:
            rows = [row[:] for row in board]
        if direction in (Direction.RIGHT, Direction.DOWN):
            rows = [row[::-1] for row in rows]

        reward = 0
        merged: list[list[int]] = []
        for row in rows:
            new_row, r = slide_and_merge_row(row)
            merged.append(new_row)
            reward += r

        if direction in (Direction.RIGHT, Direction.DOWN):
            merged = [row[::-1] for row in merged]
        if direction in (Direction.UP, Direction.DOWN):
            merged = [list(col) for col in zip(*merged)]
        return merged, reward

    def _random_spawn(self) -> None:
        """The standard spawn, used for the two starting tiles in every mode."""
        empty = [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i][j] == 0]
        if empty:
            i, j = self._rng.choice(empty)
            self.board[i][j] = 4 if self._rng.random() < self.SPAWN_FOUR_PROBABILITY else 2

    def _spawn(self) -> None:
        empty = [
            (i, j)
            for i in range(self.size)
            for j in range(self.size)
            if self.board[i][j] == 0
        ]
        if not empty:
            return
        if self.spawn_mode in ("random", "choose"):
            i, j = self._rng.choice(empty)
            value = 4 if self._rng.random() < self.SPAWN_FOUR_PROBABILITY else 2
        elif self.spawn_mode == "kind":
            i, j, value = self._kind_spawn(empty)
        else:
            i, j, value = self._extreme_spawn(empty, best=self.spawn_mode == "best")
        self.board[i][j] = value

    def _kind_spawn(self, empty: list[tuple[int, int]]) -> tuple[int, int, int]:
        """Usual 2/4 draw and uniform cell, restricted to placements that leave a
        legal move; the other value is tried before giving up."""
        value = 4 if self._rng.random() < self.SPAWN_FOUR_PROBABILITY else 2
        for v in (value, 6 - value):
            safe = [(i, j) for i, j in empty if not self._dead_after(i, j, v)]
            if safe:
                i, j = self._rng.choice(safe)
                return i, j, v
        i, j = self._rng.choice(empty)
        return i, j, value

    def _extreme_spawn(self, empty: list[tuple[int, int]], best: bool) -> tuple[int, int, int]:
        scored = [(self._reply_value(i, j, v), (i, j, v)) for i, j in empty for v in (2, 4)]
        target = max(s for s, _ in scored) if best else min(s for s, _ in scored)
        return self._rng.choice([c for s, c in scored if s == target])

    def _placed(self, i: int, j: int, value: int) -> list[list[int]]:
        board = [row[:] for row in self.board]
        board[i][j] = value
        return board

    def _dead_after(self, i: int, j: int, value: int) -> bool:
        board = self._placed(i, j, value)
        return all(self._apply(board, d)[0] == board for d in Direction)

    def _reply_value(self, i: int, j: int, value: int) -> float:
        """Best reward + evaluator(afterstate) the player can get after this spawn;
        -inf when the spawn leaves no legal move."""
        board = self._placed(i, j, value)
        best = -math.inf
        for d in Direction:
            after, reward = self._apply(board, d)
            if after != board:
                best = max(best, reward + self.evaluator(after))
        return best
