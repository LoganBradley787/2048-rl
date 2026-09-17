"""n-tuple network 2048 player: ctypes wrapper over native/ntuple.c.

The C library is compiled with the system C compiler the first time it is
needed (or when the source is newer than the built library).

    net = NTupleNet()                # 4 x 6-tuples, TC learning
    net.train(games=10_000, threads=12)
    net.play(games=20, depth=2)      # expectimax, 2 spawn layers deep
    net.save("checkpoints/ntuple/latest.bin")
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from .core import Direction, Game

NATIVE_DIR = Path(__file__).resolve().parent / "native"
SRC = NATIVE_DIR / "ntuple.c"
LIB = NATIVE_DIR / ("libntuple.dylib" if sys.platform == "darwin" else "libntuple.so")
_CKPT = Path(__file__).resolve().parent.parent / "checkpoints" / "ntuple"
DEFAULT_CANDIDATES = [_CKPT / "weights.bin", _CKPT / "latest.bin"]
_COOL_CKPT = _CKPT.parent / "ntuple_choose"
_BEST_CKPT = _CKPT.parent / "ntuple_best"     # best-scoring snapshot promoted by the snapshot loop
COOL_CANDIDATES = [_BEST_CKPT / "weights.bin", _COOL_CKPT / "weights.bin", _COOL_CKPT / "latest.bin"]
COOL_DEFAULTS = {"width": 32, "depth": 12, "stride": 4, "snake": 1.0}


def resolve_weights(path) -> Path:
    """Explicit path, else $GAME2048_NTUPLE, else the first default file that exists."""
    if path:
        return Path(path)
    env = os.environ.get("GAME2048_NTUPLE")
    if env:
        return Path(env)
    for cand in DEFAULT_CANDIDATES:
        if Path(cand).exists():
            return Path(cand)
    raise FileNotFoundError(
        f"no n-tuple weights at {DEFAULT_CANDIDATES[0]}; train some with `uv run python -m game2048.ntuple_train`"
    )


def resolve_cool_weights(path=None) -> Path:
    """Cool-mode weights: explicit path, else $GAME2048_NTUPLE_COOL, else checkpoints/ntuple_choose."""
    if path:
        return Path(path)
    env = os.environ.get("GAME2048_NTUPLE_COOL")
    if env:
        return Path(env)
    for cand in COOL_CANDIDATES:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"no cool-mode weights at {COOL_CANDIDATES[0]}; train some with "
        "`uv run python -m game2048.ntuple_train --choose --out checkpoints/ntuple_choose`"
    )

# Four 6-cell patterns (Yeh et al. 2016), cell = row * 4 + col:
#   0  1  2  3
#   4  5  6  7
#   8  9 10 11
#  12 13 14 15
DEFAULT_PATTERNS = [
    [0, 1, 2, 3, 4, 5],
    [4, 5, 6, 7, 8, 9],
    [0, 1, 2, 4, 5, 6],
    [4, 5, 6, 8, 9, 10],
]
# Eight 6-cell patterns whose symmetry orbits are all distinct: the four above plus
# a row with two cells below its middle, a corner triangle, a staircase, and the
# second row with two cells below its middle. 64 MB of weights per pattern.
PATTERN_SETS = {
    "4x6": DEFAULT_PATTERNS,
    "8x6": DEFAULT_PATTERNS + [
        [0, 1, 2, 3, 5, 6],
        [0, 1, 2, 4, 5, 8],
        [0, 1, 5, 6, 10, 11],
        [4, 5, 6, 7, 9, 10],
    ],
}
MAX_PATTERNS, MAX_LEN, MAX_STAGES = 16, 8, 8   # MAX_LEN is the C struct stride; at most MAX_CELLS cells are usable
MAX_CELLS = 7                                  # 18**7 entries is the most a 32-bit table index can hold
CELL_BITS, CELL_MASK, MAX_EXP = 5, 31, 17   # five bits per cell: tiles up to 131072 (2**17)


def cell(bits: int, i: int) -> int:
    """Exponent in cell i (row-major 0..15) of a bitboard."""
    return (bits >> (CELL_BITS * i)) & CELL_MASK


def with_cell(bits: int, i: int, exp: int) -> int:
    return (bits & ~(CELL_MASK << (CELL_BITS * i))) | (int(exp) << (CELL_BITS * i))


def _lohi(bits: int) -> tuple[int, int]:
    """A bitboard crosses the C API as two 64-bit halves."""
    return bits & 0xFFFFFFFFFFFFFFFF, (bits >> 64) & 0xFFFFFFFFFFFFFFFF


def sym_cell(cell: int, k: int) -> int:
    """Where cell index (row*4+col) lands under symmetry k in 0..7; mirrors the C code."""
    r, c = divmod(cell, 4)
    if k & 4:
        c = 3 - c
    for _ in range(k & 3):
        r, c = 3 - c, r
    return r * 4 + c

_lib = None


def _build() -> None:
    cc = os.environ.get("CC", "cc")
    tmp = LIB.with_suffix(LIB.suffix + ".tmp")
    subprocess.run([cc, "-O3", "-shared", "-fPIC", "-o", str(tmp), str(SRC), "-lpthread", "-lm"], check=True)
    os.replace(tmp, LIB)  # atomic, so a process that has the old library mapped keeps it


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    if not LIB.exists() or LIB.stat().st_mtime < SRC.stat().st_mtime:
        _build()
    L = ctypes.CDLL(str(LIB))
    u64, i32, i64, f32, f64, vp, cp = (ctypes.c_uint64, ctypes.c_int, ctypes.c_int64, ctypes.c_float,
                                        ctypes.c_double, ctypes.c_void_p, ctypes.c_char_p)
    P = ctypes.POINTER
    L.nt_create.argtypes, L.nt_create.restype = [P(i32), P(i32), i32, i32, P(i64), i32], vp
    L.nt_create_ex.argtypes, L.nt_create_ex.restype = [P(i32), P(i32), i32, i32, P(i64), i32], vp
    L.nt_tc_stages.argtypes, L.nt_tc_stages.restype = [vp], i32
    L.nt_free.argtypes = [vp]
    L.nt_num_patterns.argtypes, L.nt_num_patterns.restype = [vp], i32
    L.nt_lock_memory.argtypes, L.nt_lock_memory.restype = [vp], i32
    L.nt_use_tc.argtypes, L.nt_use_tc.restype = [vp], i32
    L.nt_num_bounds.argtypes, L.nt_num_bounds.restype = [vp], i32
    L.nt_bounds.argtypes = [vp, P(i64)]
    L.nt_stage_of.argtypes, L.nt_stage_of.restype = [u64, u64, P(i64), i32], i32
    L.nt_value_stage.argtypes, L.nt_value_stage.restype = [vp, u64, u64, i32], f32
    L.nt_update_stage.argtypes = [vp, u64, u64, f32, f32, f32, i32]
    L.nt_pattern.argtypes, L.nt_pattern.restype = [vp, i32, P(i32)], i32
    L.nt_move.argtypes, L.nt_move.restype = [u64, u64, i32, P(i32), P(u64), P(u64)], None
    L.nt_spawn.argtypes, L.nt_spawn.restype = [u64, u64, P(u64), P(u64), P(u64)], None
    L.nt_cell_bits.restype, L.nt_max_exp.restype = i32, i32
    L.nt_value.argtypes, L.nt_value.restype = [vp, u64, u64], f32
    L.nt_update.argtypes = [vp, u64, u64, f32, f32, f32]
    L.nt_train.argtypes = [vp, ctypes.c_long, i32, f32, f32, u64, P(i64), P(i32), P(i32)]
    L.nt_best_move.argtypes, L.nt_best_move.restype = [vp, u64, u64, i32, f64], i32
    L.nt_search_value.argtypes, L.nt_search_value.restype = [vp, u64, u64, i32, f64], f64
    L.nt_play.argtypes = [vp, ctypes.c_long, i32, f64, i32, u64, P(i64), P(i32), P(i32)]
    L.nt_best_choose.argtypes, L.nt_best_choose.restype = [vp, u64, u64, i32, i32, P(i32), P(i32)], i32
    L.nt_choose_value.argtypes, L.nt_choose_value.restype = [vp, u64, u64, i32, i32], f64
    L.nt_play_choose.argtypes = [vp, ctypes.c_long, i32, i32, i32, u64, ctypes.c_long, ctypes.c_long,
                                 P(i64), P(i32), P(i32)]
    L.nt_train_choose.argtypes = [vp, ctypes.c_long, i32, f32, f32, u64, i32, i32, ctypes.c_long, f64,
                                  P(u64), ctypes.c_long, f64, P(i64), P(i32), P(i32)]
    L.nt_beam_choose.argtypes, L.nt_beam_choose.restype = [vp, u64, u64, i32, i32, i32, f64, P(i32), P(i32)], i32
    L.nt_snake_score.argtypes, L.nt_snake_score.restype = [u64, u64], f64
    L.nt_beam_value.argtypes, L.nt_beam_value.restype = [vp, u64, u64, i32, i32, i32, f64], f64
    L.nt_beam_plan.argtypes, L.nt_beam_plan.restype = [vp, u64, u64, i32, i32, i32, f64, P(i32), P(i32), P(i32)], i32
    L.nt_play_beam.argtypes = [vp, ctypes.c_long, i32, i32, i32, i32, f64, i32, u64, ctypes.c_long, ctypes.c_long,
                               P(i64), P(i32), P(i32)]
    L.nt_save.argtypes, L.nt_save.restype = [vp, cp], i32
    L.nt_save_ex.argtypes, L.nt_save_ex.restype = [vp, cp, i32], i32
    L.nt_load.argtypes, L.nt_load.restype = [cp], vp
    L.nt_load_ex.argtypes, L.nt_load_ex.restype = [cp, P(i64), i32, i32], vp
    _lib = L
    return L


# --- board helpers --------------------------------------------------------------

def to_bits(board) -> int:
    """(4, 4) exponents -> bitboard (a Python int, five bits per cell). A board of
    tile values whose largest tile is above 17 is converted too, but a Game board
    must use `tiles_to_bits`: early boards of 2s, 4s and 8s look like exponents."""
    arr = np.asarray(board)
    if arr.max() > MAX_EXP:  # tile values, not exponents
        return tiles_to_bits(arr)
    bits = 0
    for i, v in enumerate(arr.reshape(16)):
        bits |= int(v) << (CELL_BITS * i)
    return bits


def tiles_to_bits(board) -> int:
    """(4, 4) tile values (0, 2, 4, ...) -> bitboard of exponents."""
    bits = 0
    for i, v in enumerate(np.asarray(board).reshape(16)):
        v = int(v)
        if v:
            bits |= (v.bit_length() - 1) << (CELL_BITS * i)
    return bits


def from_bits(bits: int) -> np.ndarray:
    return np.array([cell(bits, i) for i in range(16)], dtype=np.uint8).reshape(4, 4)


def move(bits: int, direction: int) -> tuple[int, int]:
    """Apply a move; returns (new_bits, reward). Unchanged bits mean an invalid move."""
    reward, lo, hi = ctypes.c_int(0), ctypes.c_uint64(0), ctypes.c_uint64(0)
    lib().nt_move(*_lohi(bits), int(direction), ctypes.byref(reward), ctypes.byref(lo), ctypes.byref(hi))
    return lo.value | (hi.value << 64), reward.value


def afterstates_valid(bits: int) -> list[int]:
    """Directions that change the board."""
    return [d for d in range(4) if move(bits, d)[0] != bits]


def spawn(bits: int, seed: int) -> int:
    state = ctypes.c_uint64((seed * 0x9E3779B97F4A7C15 + 0x1234567) & 0xFFFFFFFFFFFFFFFF)
    lo, hi = ctypes.c_uint64(0), ctypes.c_uint64(0)
    lib().nt_spawn(*_lohi(bits), ctypes.byref(state), ctypes.byref(lo), ctypes.byref(hi))
    return lo.value | (hi.value << 64)


def board_mass(bits: int) -> int:
    """Sum of all tiles on the board."""
    return sum(1 << cell(bits, i) for i in range(16) if cell(bits, i))


def harvest_starts(net, games: int = 1, min_mass: int = 40_000, every: int = 200, width: int = 32,
                   depth: int = 12, stride: int = 4, snake: float = 0.0, seed: int = 0,
                   max_moves: int = 200_000) -> np.ndarray:
    """Play the choose game with the beam and record a board every `every` moves
    once the tile mass reaches `min_mass`. Returns (n, 4, 4) uint8 exponent grids
    to feed back to training as late-game starts (see `train_choose(starts=...)`)."""
    grids = []
    for g in range(games):
        b = spawn(spawn(0, seed * 7919 + g), seed * 7919 + g + 1)
        moves, plan = 0, []
        while moves < max_moves:
            if not plan:
                plan = net.beam_plan(b, width, depth, 0, snake)[: max(1, stride)]
            if not plan:
                break
            m, c, v = plan.pop(0)
            b, _ = move(b, m)
            b = with_cell(b, c, v)
            moves += 1
            if moves % every == 0 and board_mass(b) >= min_mass:
                grids.append(from_bits(b))
    return np.array(grids, dtype=np.uint8).reshape(-1, 4, 4)


def save_starts(path, grids) -> None:
    np.save(path, np.asarray(grids, dtype=np.uint8).reshape(-1, 4, 4))


def load_starts(path) -> list[int]:
    """Saved start grids -> list of bitboards."""
    return [to_bits(g) for g in np.load(path)]


def cool_search_settings(weights) -> dict:
    """Beam settings for cool-mode play: `best.json` beside the weights when the
    snapshot loop wrote one (the config that scored best on those tables), else
    the defaults. Keys: width, depth, stride, snake."""
    settings = dict(COOL_DEFAULTS)
    meta = Path(weights).with_name("best.json")
    if meta.exists():
        saved = json.loads(meta.read_text())
        settings.update({k: saved[k] for k in settings if k in saved})
    return settings


def snake_score(bits: int) -> float:
    """Tiles read along the snake path (0 1 2 3 / 7 6 5 4 / ...), weighted 0.5**k,
    best of the 8 symmetries. A chain laid out as a snake scores highest."""
    return float(lib().nt_snake_score(*_lohi(bits)))


def stage_of(bits: int, boundaries: list[int]) -> int:
    """Stage index: how many mass boundaries (sum of all tiles) the board has reached."""
    arr = (ctypes.c_int64 * max(1, len(boundaries)))(*boundaries)
    return int(lib().nt_stage_of(*_lohi(bits), arr, len(boundaries)))


def _stats(scores, maxtiles, moves) -> dict:
    s, t, m = (np.ctypeslib.as_array(x).copy() for x in (scores, maxtiles, moves))
    return {"scores": s, "max_tiles": t, "moves": m}


# --- network --------------------------------------------------------------------

class NTupleNet:
    """`boundaries` are tile-mass thresholds (sum of all tiles on the board) that
    switch to a fresh set of weight tables, e.g. [16384, 24576, 32768]. Entries a
    later stage has never updated read through to the previous stage.

    `tc_stages` is how many leading stages learn with temporal coherence (per-entry
    adaptive rates, 8 extra bytes per weight); the rest use plain TD with their own
    step size (`alpha_plain`). Default: all stages when `tc`, else none."""

    def __init__(self, patterns=None, tc: bool = True, boundaries=None, tc_stages: int | None = None,
                 _handle=None) -> None:
        self._lib = lib()
        if _handle is not None:
            self._h = _handle
        else:
            patterns = [list(p) for p in (patterns or DEFAULT_PATTERNS)]
            if not 1 <= len(patterns) <= MAX_PATTERNS or any(not 1 <= len(p) <= MAX_CELLS for p in patterns):
                raise ValueError("1..16 patterns of 1..7 cells each")
            boundaries = sorted(int(b) for b in (boundaries or []))
            if len(boundaries) >= MAX_STAGES:
                raise ValueError(f"at most {MAX_STAGES - 1} boundaries")
            flat = (ctypes.c_int * (MAX_PATTERNS * MAX_LEN))()
            lens = (ctypes.c_int * MAX_PATTERNS)()
            for k, p in enumerate(patterns):
                lens[k] = len(p)
                for j, c in enumerate(p):
                    flat[k * MAX_LEN + j] = int(c)
            barr = (ctypes.c_int64 * MAX_STAGES)(*boundaries)
            if tc_stages is None:
                tc_stages = len(boundaries) + 1 if tc else 0
            self._h = self._lib.nt_create_ex(flat, lens, len(patterns), int(tc_stages), barr, len(boundaries))
            if not self._h:
                raise RuntimeError("could not create n-tuple network")

    @classmethod
    def load(cls, path, boundaries=None, keep_tc: bool = True, tc_stages: int | None = None) -> "NTupleNet":
        """With `boundaries`, a single-stage file becomes stage 0 of a multi-stage
        network (later stages start by reading through to it). A file that already
        has stages must be loaded with no boundaries or the same ones. keep_tc=False
        drops the temporal-coherence state everywhere; `tc_stages=n` keeps it on the
        first n stages only (plain TD elsewhere, at a third of the memory per stage)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        bl = sorted(int(b) for b in (boundaries or []))
        arr = (ctypes.c_int64 * max(1, len(bl)))(*bl)
        tc_req = int(tc_stages) if tc_stages is not None else (-1 if keep_tc else 0)
        h = lib().nt_load_ex(str(path).encode(), arr, len(bl), tc_req)
        if not h:
            raise ValueError(f"{path} could not be loaded" + (f" with boundaries {bl}" if bl else "") +
                             ": not a valid weights file, or a staged file with different boundaries")
        return cls(_handle=h)

    def save(self, path, weights_only: bool = False) -> None:
        """weights_only drops the learning state (cannot resume training from it)
        and bakes stage promotion in; a third of the size, identical play."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self._lib.nt_save_ex(self._h, str(path).encode(), int(weights_only)) != 0:
            raise OSError(f"could not write {path}")

    def close(self) -> None:
        if getattr(self, "_h", None):
            self._lib.nt_free(self._h)
            self._h = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def lock_memory(self) -> bool:
        """Pin the tables in RAM (mlock). Returns False if the OS refused."""
        return self._lib.nt_lock_memory(self._h) == 0

    @property
    def patterns(self) -> list[list[int]]:
        out = []
        buf = (ctypes.c_int * MAX_LEN)()
        for k in range(self._lib.nt_num_patterns(self._h)):
            n = self._lib.nt_pattern(self._h, k, buf)
            out.append([buf[j] for j in range(n)])
        return out

    @property
    def tc(self) -> bool:
        return bool(self._lib.nt_use_tc(self._h))

    @property
    def tc_stages(self) -> int:
        return int(self._lib.nt_tc_stages(self._h))

    @property
    def boundaries(self) -> list[int]:
        n = self._lib.nt_num_bounds(self._h)
        buf = (ctypes.c_int64 * MAX_STAGES)()
        self._lib.nt_bounds(self._h, buf)
        return [int(buf[i]) for i in range(n)]

    def value(self, bits: int) -> float:
        return float(self._lib.nt_value(self._h, *_lohi(bits)))

    def value_in_stage(self, bits: int, stage: int) -> float:
        return float(self._lib.nt_value_stage(self._h, *_lohi(bits), int(stage)))

    def update(self, bits: int, delta: float, alpha: float = 1.0, alpha_plain: float | None = None) -> None:
        """`alpha` is the step for temporal-coherence stages (1.0 is standard), `alpha_plain`
        the total step for plain-TD stages (defaults to `alpha`)."""
        ap = alpha if alpha_plain is None else alpha_plain
        self._lib.nt_update(self._h, *_lohi(bits), float(delta), float(alpha), float(ap))

    def update_in_stage(self, bits: int, delta: float, alpha: float = 1.0, stage: int = -1,
                        alpha_plain: float | None = None) -> None:
        ap = alpha if alpha_plain is None else alpha_plain
        self._lib.nt_update_stage(self._h, *_lohi(bits), float(delta), float(alpha), float(ap), int(stage))

    def train(self, games: int, threads: int = 1, alpha: float = 1.0, seed: int = 0,
              alpha_plain: float | None = None) -> dict:
        """Play `games` self-play games with TD(0) learning; returns per-game stats."""
        ap = alpha if alpha_plain is None else alpha_plain
        scores = (ctypes.c_int64 * games)()
        maxtiles = (ctypes.c_int32 * games)()
        moves = (ctypes.c_int32 * games)()
        self._lib.nt_train(self._h, games, threads, float(alpha), float(ap), seed & 0xFFFFFFFFFFFFFFFF,
                           scores, maxtiles, moves)
        return _stats(scores, maxtiles, moves)

    def best_move(self, bits: int, depth: int = 1, cutoff: float = 0.0) -> int:
        """Expectimax move for the board, -1 if none is legal. depth = spawn layers searched."""
        return int(self._lib.nt_best_move(self._h, *_lohi(bits), int(depth), float(cutoff)))

    def search_value(self, bits: int, depth: int = 1, cutoff: float = 0.0) -> float:
        return float(self._lib.nt_search_value(self._h, *_lohi(bits), int(depth), float(cutoff)))

    # --- choose mode: the player places every tile ---------------------------

    def choose_value(self, bits: int, depth: int = 1, topk: int = 4) -> float:
        """Best achievable reward + value when the player also places the tiles;
        `depth` placement decisions searched, `topk` placements kept per node."""
        return float(self._lib.nt_choose_value(self._h, *_lohi(bits), int(depth), int(topk)))

    def best_choose(self, bits: int, depth: int = 2, topk: int = 4) -> tuple[int, int, int]:
        """(move, cell 0..15, tile exponent 1 or 2); move is -1 when nothing is legal."""
        cell, value = ctypes.c_int(-1), ctypes.c_int(0)
        m = int(self._lib.nt_best_choose(self._h, *_lohi(bits), int(depth), int(topk), ctypes.byref(cell), ctypes.byref(value)))
        return m, cell.value, value.value

    def play_choose(self, games: int, depth: int = 2, topk: int = 4, threads: int = 1, seed: int = 0,
                    max_moves: int = 200_000, prefix: int = 0) -> dict:
        """Play the choose game with the max-max tree. The first `prefix` moves get
        random (never immediately fatal) spawns instead of the chosen tile, so the
        games differ: the choose game is otherwise deterministic, and a deterministic
        policy funnels different starts into the same line within a few moves."""
        scores = (ctypes.c_int64 * games)()
        maxtiles = (ctypes.c_int32 * games)()
        moves = (ctypes.c_int32 * games)()
        self._lib.nt_play_choose(self._h, games, int(depth), int(topk), threads, seed & 0xFFFFFFFFFFFFFFFF,
                                 int(max_moves), int(prefix), scores, maxtiles, moves)
        return _stats(scores, maxtiles, moves)

    # --- beam search: plan a line of the deterministic choose game ------------

    def beam_value(self, bits: int, width: int = 64, depth: int = 16, spread: int = 0, snake: float = 0.0) -> float:
        """Value of the best line found by a beam of `width` boards over `depth`
        move+placement steps; unlimited width equals the full tree of that depth.
        `spread` > 0 caps the survivors per parent so the beam covers distinct lines;
        `snake` adds that weight times `snake_score` to every leaf."""
        return float(self._lib.nt_beam_value(self._h, *_lohi(bits), int(width), int(depth), int(spread), float(snake)))

    def beam_choose(self, bits: int, width: int = 64, depth: int = 16, spread: int = 0,
                    snake: float = 0.0) -> tuple[int, int, int]:
        """(move, cell 0..15, tile exponent 1 or 2) starting the best line; move -1 if stuck."""
        cell_, value = ctypes.c_int(-1), ctypes.c_int(0)
        m = int(self._lib.nt_beam_choose(self._h, *_lohi(bits), int(width), int(depth), int(spread), float(snake),
                                         ctypes.byref(cell_), ctypes.byref(value)))
        return m, cell_.value, value.value

    def beam_plan(self, bits: int, width: int = 64, depth: int = 16, spread: int = 0,
                  snake: float = 0.0) -> list[tuple[int, int, int]]:
        """The best line found: [(move, cell, tile exponent), ...], empty when stuck."""
        depth = int(depth)
        moves, cells, values = (ctypes.c_int * depth)(), (ctypes.c_int * depth)(), (ctypes.c_int * depth)()
        n = int(self._lib.nt_beam_plan(self._h, *_lohi(bits), int(width), depth, int(spread), float(snake),
                                       moves, cells, values))
        return [(moves[i], cells[i], values[i]) for i in range(n)]

    def play_beam(self, games: int, width: int = 64, depth: int = 16, threads: int = 1, seed: int = 0,
                  max_moves: int = 200_000, stride: int = 1, spread: int = 0, prefix: int = 0,
                  snake: float = 0.0) -> dict:
        """Play with beam search; `stride` steps of each plan are committed before
        re-planning; the first `prefix` moves get random spawns (see play_choose)."""
        scores = (ctypes.c_int64 * games)()
        maxtiles = (ctypes.c_int32 * games)()
        moves = (ctypes.c_int32 * games)()
        self._lib.nt_play_beam(self._h, games, int(width), int(depth), int(stride), int(spread), float(snake), threads,
                               seed & 0xFFFFFFFFFFFFFFFF, int(max_moves), int(prefix), scores, maxtiles, moves)
        return _stats(scores, maxtiles, moves)

    def train_choose(self, games: int, threads: int = 1, alpha: float = 1.0, seed: int = 0,
                     max_moves: int = 200_000, alpha_plain: float | None = None, depth: int = 1,
                     topk: int = 4, explore: float = 0.0, starts=None, start_frac: float = 0.0) -> dict:
        """Self-play TD(0) learning of the choose game: the agent places every tile,
        picking move and placement with the choose search at `depth`. Games are cut
        at `max_moves` (without a terminal update). With probability `explore` a step
        places a random tile instead, so games leave the one line a deterministic
        policy replays. `starts` is a list of bitboards; a `start_frac` share of the
        games begins from one of them instead of two tiles (late-game restarts, so
        the endgame gets trained more than once per 30k-move game). Returns per-game stats."""
        ap = alpha if alpha_plain is None else alpha_plain
        scores = (ctypes.c_int64 * games)()
        maxtiles = (ctypes.c_int32 * games)()
        moves = (ctypes.c_int32 * games)()
        starts = list(starts or [])
        packed = (ctypes.c_uint64 * max(2, 2 * len(starts)))(*[h for b in starts for h in _lohi(b)])
        self._lib.nt_train_choose(self._h, games, threads, float(alpha), float(ap), seed & 0xFFFFFFFFFFFFFFFF,
                                  int(depth), int(topk), int(max_moves), float(explore),
                                  packed, len(starts), float(start_frac), scores, maxtiles, moves)
        return _stats(scores, maxtiles, moves)

    def play(self, games: int, depth: int = 1, cutoff: float = 0.0, threads: int = 1, seed: int = 0) -> dict:
        scores = (ctypes.c_int64 * games)()
        maxtiles = (ctypes.c_int32 * games)()
        moves = (ctypes.c_int32 * games)()
        self._lib.nt_play(self._h, games, int(depth), float(cutoff), threads, seed & 0xFFFFFFFFFFFFFFFF,
                          scores, maxtiles, moves)
        return _stats(scores, maxtiles, moves)


def summarize(stats: dict) -> dict:
    s, t = stats["scores"], stats["max_tiles"]
    uniq, counts = np.unique(t, return_counts=True)
    return {
        "games": int(len(s)),
        "mean_score": float(s.mean()),
        "median_score": float(np.median(s)),
        "max_score": int(s.max()),
        "mean_length": float(stats["moves"].mean()),
        "max_tile": int(t.max()),
        "tile_counts": {int(a): int(b) for a, b in zip(uniq, counts)},
        **{f"rate_{v}": float((t >= v).mean()) for v in (1024, 2048, 4096, 8192, 16384, 32768)},
    }


class NTupleAgent:
    """Plays with a trained n-tuple network plus expectimax. Registered as "ntuple";
    "ntuple-cool" is the same class on tables trained for cool mode."""

    def __init__(self, weights=None, depth: int = 3, cutoff: float = 0.0, choose_depth: int = 3,
                 topk: int = 4, beam_width: int = 0, beam_depth: int = 16, beam_stride: int = 1,
                 beam_spread: int = 0, beam_snake: float = 0.0) -> None:
        path = resolve_weights(weights)
        if not path.exists():
            raise FileNotFoundError(
                f"no n-tuple weights at {path}; train some with `uv run python -m game2048.ntuple_train`"
            )
        self.net = NTupleNet.load(path)
        self.depth = depth
        self.cutoff = cutoff
        self.choose_depth = choose_depth
        self.topk = topk
        self.beam_width = beam_width      # > 0: plan with a beam instead of the max-max tree
        self.beam_depth = beam_depth
        self.beam_stride = beam_stride    # steps of a plan to follow before planning again
        self.beam_spread = beam_spread    # survivors per parent (0 = unlimited)
        self.beam_snake = beam_snake      # weight of the snake-order bonus at the leaves
        self._plan: list[tuple[int, int, int]] = []
        self._plan_board = -1             # board the next cached step applies to

    def _beam_step(self, bits: int) -> tuple[int, int, int]:
        if not self._plan or self._plan_board != bits:
            self._plan = self.net.beam_plan(bits, self.beam_width, self.beam_depth, self.beam_spread,
                                            self.beam_snake)[: max(1, self.beam_stride)]
        if not self._plan:
            return -1, -1, 0
        m, cell, value = self._plan.pop(0)
        after, _ = move(bits, m)
        self._plan_board = with_cell(after, cell, value)   # the plan continues from this board
        return m, cell, value

    def choose(self, game: Game) -> tuple[Direction, tuple[int, int, int]]:
        """Choose mode: the move and the (row, col, 2|4) placement, searched jointly."""
        bits = tiles_to_bits(game.board)
        if self.beam_width > 0:
            m, cell, value = self._beam_step(bits)
        else:
            m, cell, value = self.net.best_choose(bits, self.choose_depth, self.topk)
        if m < 0:
            raise ValueError("no legal moves")
        return Direction(m), (cell // 4, cell % 4, 2 ** value)

    def act(self, game: Game) -> Direction:
        d = self.net.best_move(tiles_to_bits(game.board), self.depth, self.cutoff)
        if d < 0:
            raise ValueError("no legal moves")
        return Direction(d)
