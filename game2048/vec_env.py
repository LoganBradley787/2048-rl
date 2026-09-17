"""Vectorised 2048 for training: N boards as a (N, 4, 4) uint8 array of
exponents (0 = empty, k = tile 2**k).

Every move is a left-slide of rows, done with a 65536-entry lookup table
indexed by the row's four nibbles. Other directions reverse/transpose first.
The merge rule is the same as `core.slide_and_merge_row`; the only addition
is a cap so two 32768 tiles never merge (2**16 does not fit in a nibble).

Action order matches `core.Direction`: UP=0, RIGHT=1, DOWN=2, LEFT=3.
"""

from __future__ import annotations

import numpy as np

MAX_EXP = 15


def _merge_exps(row: list[int]) -> tuple[list[int], int]:
    tiles = [v for v in row if v]
    out: list[int] = []
    reward = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1] and tiles[i] < MAX_EXP:
            out.append(tiles[i] + 1)
            reward += 2 ** (tiles[i] + 1)
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    return out + [0] * (len(row) - len(out)), reward


def _build_tables() -> tuple[np.ndarray, np.ndarray]:
    row_lut = np.zeros(65536, dtype=np.uint16)
    reward_lut = np.zeros(65536, dtype=np.int64)
    for idx in range(65536):
        row = [(idx >> 12) & 15, (idx >> 8) & 15, (idx >> 4) & 15, idx & 15]
        out, reward = _merge_exps(row)
        row_lut[idx] = (out[0] << 12) | (out[1] << 8) | (out[2] << 4) | out[3]
        reward_lut[idx] = reward
    return row_lut, reward_lut


ROW_LUT, REWARD_LUT = _build_tables()


def encode_rows(boards: np.ndarray) -> np.ndarray:
    b = boards.astype(np.uint16)
    return (b[..., 0] << 12) | (b[..., 1] << 8) | (b[..., 2] << 4) | b[..., 3]


def decode_rows(idx: np.ndarray) -> np.ndarray:
    return np.stack(
        [(idx >> 12) & 15, (idx >> 8) & 15, (idx >> 4) & 15, idx & 15], axis=-1
    ).astype(np.uint8)


def move_left(boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = encode_rows(boards)
    return decode_rows(ROW_LUT[idx]), REWARD_LUT[idx].sum(axis=-1)


def afterstates(boards: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All four moves at once.

    Returns after (N, 4, 4, 4) indexed [board, action, row, col], rewards
    (N, 4) and valid (N, 4) meaning the move changes the board.
    """
    left, r_left = move_left(boards)
    right, r_right = move_left(boards[:, :, ::-1])
    right = right[:, :, ::-1]
    t = boards.transpose(0, 2, 1)
    up, r_up = move_left(t)
    up = up.transpose(0, 2, 1)
    down, r_down = move_left(t[:, :, ::-1])
    down = down[:, :, ::-1].transpose(0, 2, 1)

    after = np.stack([up, right, down, left], axis=1)
    rewards = np.stack([r_up, r_right, r_down, r_left], axis=1)
    valid = (after != boards[:, None]).any(axis=(2, 3))
    return np.ascontiguousarray(after), rewards, valid


def _pick_one(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One uniformly chosen True per row of `mask` (rows with none stay all False)."""
    counts = mask.sum(axis=1)
    pick = (rng.random(len(mask)) * counts).astype(np.int64)
    return mask & (mask.cumsum(axis=1) == pick[:, None] + 1)


def _safe_cells(boards: np.ndarray, value: int) -> np.ndarray:
    """(N, 16) mask: empty cells where placing `value` leaves a legal move."""
    n = len(boards)
    cand = np.repeat(boards[:, None], 16, axis=1).reshape(n, 16, 16)
    idx = np.arange(16)
    empty = boards.reshape(n, 16) == 0
    cand[:, idx, idx] = value
    alive = afterstates(cand.reshape(-1, 4, 4))[2].any(axis=1).reshape(n, 16)
    return alive & empty


def spawn(boards: np.ndarray, rng: np.random.Generator, kind: bool = False) -> None:
    """In place: add a 2 (p=0.9) or 4 (p=0.1) to a random empty cell of every
    board that has one. With `kind`, placements that would end the game are
    avoided while any placement keeps a move (the other value is tried before
    giving up)."""
    n = len(boards)
    flat = boards.reshape(n, 16)
    empty = flat == 0
    vals = np.where(rng.random(n) < 0.1, 2, 1).astype(np.uint8)
    if not kind:
        target = _pick_one(empty, rng)
        flat[target] = vals[target.any(axis=1)]
        return
    safe = {1: _safe_cells(boards, 1), 2: _safe_cells(boards, 2)}
    primary = np.where(vals[:, None] == 1, safe[1], safe[2])
    secondary = np.where(vals[:, None] == 1, safe[2], safe[1])
    target = _pick_one(primary, rng)
    done = target.any(axis=1)
    t2 = _pick_one(secondary, rng)
    use2 = ~done & t2.any(axis=1)
    target[use2] = t2[use2]
    vals[use2] = 3 - vals[use2]
    done |= use2
    t3 = _pick_one(empty, rng)
    target[~done] = t3[~done]
    flat[target] = vals[target.any(axis=1)]


# --- symmetries (dihedral group of the square) -------------------------------

def apply_symmetry(boards: np.ndarray, k: int) -> np.ndarray:
    """k in 0..7: rotations 0-3, then the same four after a horizontal flip."""
    b = boards[:, :, ::-1] if k >= 4 else boards
    return np.ascontiguousarray(np.rot90(b, k % 4, axes=(1, 2)))


def apply_inverse_symmetry(boards: np.ndarray, k: int) -> np.ndarray:
    b = np.rot90(boards, -(k % 4), axes=(1, 2))
    if k >= 4:
        b = b[:, :, ::-1]
    return np.ascontiguousarray(b)


# --- batched environment -----------------------------------------------------

class VecEnv:
    def __init__(self, n: int, seed: int | None = None, spawn_mode: str = "random") -> None:
        if spawn_mode not in ("random", "kind"):
            raise NotImplementedError("VecEnv supports spawn modes 'random' and 'kind'; "
                                      "use core.Game for 'best' and 'evil'")
        self.n = n
        self.spawn_mode = spawn_mode
        self.rng = np.random.default_rng(seed)
        self.boards = np.zeros((n, 4, 4), dtype=np.uint8)
        self.scores = np.zeros(n, dtype=np.int64)
        self.reset(np.ones(n, dtype=bool))

    def reset(self, mask: np.ndarray) -> None:
        if not mask.any():
            return
        fresh = np.zeros((int(mask.sum()), 4, 4), dtype=np.uint8)
        spawn(fresh, self.rng, kind=self.spawn_mode == "kind")
        spawn(fresh, self.rng, kind=self.spawn_mode == "kind")
        self.boards[mask] = fresh
        self.scores[mask] = 0

    def legal_mask(self) -> np.ndarray:
        return afterstates(self.boards)[2]

    def done(self) -> np.ndarray:
        return ~self.legal_mask().any(axis=1)

    def apply(self, actions: np.ndarray, after: np.ndarray, rewards: np.ndarray,
              valid: np.ndarray) -> np.ndarray:
        """Move every board by its action using precomputed afterstates.
        Invalid actions leave the board alone. Returns the rewards."""
        idx = np.arange(self.n)
        ok = valid[idx, actions]
        r = np.where(ok, rewards[idx, actions], 0)
        new = np.where(ok[:, None, None], after[idx, actions], self.boards)
        moved = new[ok]
        spawn(moved, self.rng, kind=self.spawn_mode == "kind")
        new[ok] = moved
        self.boards = np.ascontiguousarray(new)
        self.scores += r
        return r

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        after, rewards, valid = afterstates(self.boards)
        r = self.apply(np.asarray(actions), after, rewards, valid)
        return r, self.done()

    def max_tiles(self) -> np.ndarray:
        return 2 ** self.boards.reshape(self.n, 16).max(axis=1).astype(np.int64)
