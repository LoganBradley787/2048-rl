"""Gym-style wrapper around `Game` for training agents.

    env = Env2048(seed=0)
    obs = env.reset()
    obs, reward, done, info = env.step(action)   # action in 0..3

Observations are the board flattened row-major, each cell encoded as
log2(tile) (0 for empty). `obs_array()` gives the same thing as a numpy array.
"""

from __future__ import annotations

from typing import Any

from .core import Direction, Game

Observation = list[int]


class Env2048:
    def __init__(
        self,
        size: int = 4,
        seed: int | None = None,
        invalid_move_penalty: float = 0.0,
        max_invalid_moves: int | None = None,
        spawn_mode: str = "random",
        evaluator=None,
    ) -> None:
        self.size = size
        self._seed = seed
        self.spawn_mode = spawn_mode
        self.evaluator = evaluator
        self.invalid_move_penalty = invalid_move_penalty
        self.max_invalid_moves = max_invalid_moves
        self.game: Game | None = None
        self._invalid_streak = 0

    @property
    def n_actions(self) -> int:
        return len(Direction)

    def reset(self) -> Observation:
        self.game = Game(size=self.size, seed=self._seed, spawn_mode=self.spawn_mode, evaluator=self.evaluator)
        self._invalid_streak = 0
        return self.observe()

    def observe(self) -> Observation:
        game = self._require_game()
        return [v.bit_length() - 1 if v else 0 for row in game.board for v in row]

    def obs_array(self):
        import numpy as np  # optional dependency, imported lazily

        return np.asarray(self.observe(), dtype=np.int64)

    def action_mask(self) -> list[bool]:
        legal = set(self._require_game().legal_moves())
        return [d in legal for d in Direction]

    def step(self, action: Direction | int) -> tuple[Observation, float, bool, dict[str, Any]]:
        game = self._require_game()
        moved, reward = game.move(action)
        if moved:
            self._invalid_streak = 0
            reward_out: float = reward
        else:
            self._invalid_streak += 1
            reward_out = -self.invalid_move_penalty

        done = game.is_over()
        if (
            self.max_invalid_moves is not None
            and self._invalid_streak >= self.max_invalid_moves
        ):
            done = True

        info = {
            "moved": moved,
            "score": game.score,
            "max_tile": game.max_tile(),
            "legal_moves": [int(d) for d in game.legal_moves()],
            "mode": game.spawn_mode,
        }
        return self.observe(), reward_out, done, info

    def _require_game(self) -> Game:
        if self.game is None:
            raise RuntimeError("call reset() before using the environment")
        return self.game
