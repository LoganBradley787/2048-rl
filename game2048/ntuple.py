"""n-tuple network 2048 player: ctypes wrapper over native/ntuple.c.

The C library is compiled with the system C compiler the first time it is
needed (or when the source or this file is newer than the built library).

    net = NTupleNet()                # 4 x 6-tuples, TC learning
    net.train(games=10_000, threads=12)
    net.play(games=20, depth=2)      # expectimax, 2 spawn layers deep
    net.save("checkpoints/ntuple/latest.bin")
"""

from __future__ import annotations

import ctypes
import json
import math
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
_BEST_CKPT = _CKPT.parent / "ntuple_best"     # a hand-picked best snapshot, with its beam settings in best.json
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
    # The eight patterns Matsuzaki selected by measured usefulness (2016), the set
    # behind the strongest published n-tuple players (Guei et al. 2022).
    "8x6-matsuzaki": [
        [0, 1, 2, 4, 5, 6],
        [4, 5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6, 9],
        [0, 1, 2, 5, 9, 10],
        [3, 4, 5, 6, 7, 8],
        [1, 3, 4, 5, 6, 7],
        [0, 1, 4, 8, 9, 10],
    ],
}
# The same eight with smaller patterns beside them (redundant encoding, Jaśkowski
# 2017): the two straight 4-cell lines, the three 2x2 squares and the two straight
# 3-cell lines that differ under symmetry. Their tables are tiny and learn fast;
# the price is twice the lookups per board.
PATTERN_SETS["8x6-matsuzaki-redundant"] = PATTERN_SETS["8x6-matsuzaki"] + [
    [0, 1, 2, 3], [4, 5, 6, 7],
    [0, 1, 4, 5], [1, 2, 5, 6], [5, 6, 9, 10],
    [0, 1, 2], [4, 5, 6],
]
MAX_PATTERNS, MAX_LEN, MAX_STAGES = 16, 8, 8   # MAX_LEN is the C struct stride; at most MAX_CELLS cells are usable
MAX_CELLS = 7                                  # 18**7 entries is the most a 32-bit table index can hold
CELL_BITS, CELL_MASK, MAX_EXP = 5, 31, 17   # five bits per cell: tiles up to 131072 (2**17)
BEAM_MAX_DEPTH = 256                           # the C beam plans at most this many steps
SNAKE_DECAY = 0.5                              # default weight ratio between consecutive snake cells


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


def _cpu_has(*wanted: str) -> bool:
    """True when /proc/cpuinfo (Linux, x86) lists every one of these CPU flags."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags"):
                return set(wanted) <= set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return False


def _build() -> None:
    cc = os.environ.get("CC", "cc")
    tmp = LIB.with_suffix(LIB.suffix + ".tmp")
    # -ffp-contract=off: no fused multiply-add, so every CPU does the same arithmetic.
    # Self-play is chaotic; with fusing on, an arm64 and an x86-64 build train different tables.
    flags = ["-O3", "-ffp-contract=off", "-shared", "-fPIC"]
    # The library is built on the machine that runs it, so it may use that CPU's
    # integer vector instructions; the engine has a vector path for indexing the
    # tables. Floating point is untouched: same results with or without.
    if os.environ.get("GAME2048_NO_SIMD") != "1" and _cpu_has("avx2", "bmi2"):
        flags += ["-mavx2", "-mbmi2"]
    try:
        subprocess.run([cc, *flags, "-o", str(tmp), str(SRC), "-lpthread", "-lm"], check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        raise ImportError(f"could not build the C engine with {cc!r} ({e}); it needs a C compiler "
                          "(Xcode command line tools on macOS, gcc or clang on Linux; set CC to pick one)") from e
    os.replace(tmp, LIB)  # atomic, so a process that has the old library mapped keeps it


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    newest = max(SRC.stat().st_mtime, Path(__file__).stat().st_mtime)   # this file holds the compiler flags
    if not LIB.exists() or LIB.stat().st_mtime < newest:
        _build()
    L = ctypes.CDLL(str(LIB))
    u64, i32, i64, f32, f64, vp, cp = (ctypes.c_uint64, ctypes.c_int, ctypes.c_int64, ctypes.c_float,
                                        ctypes.c_double, ctypes.c_void_p, ctypes.c_char_p)
    P = ctypes.POINTER
    L.nt_create_ex.argtypes, L.nt_create_ex.restype = [P(i32), P(i32), i32, i32, P(i64), i32], vp
    L.nt_tc_stages.argtypes, L.nt_tc_stages.restype = [vp], i32
    L.nt_free.argtypes = [vp]
    L.nt_fill.argtypes = [vp, f32]
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
    L.nt_value.argtypes, L.nt_value.restype = [vp, u64, u64], f32
    L.nt_update.argtypes = [vp, u64, u64, f32, f32, f32]
    L.nt_train.argtypes = [vp, ctypes.c_long, i32, f32, f32, u64, P(i64), P(i32), P(i32)]
    L.nt_train_from.argtypes = [vp, ctypes.c_long, i32, f32, f32, u64, P(u64), ctypes.c_long, P(i64), P(i32), P(i32)]
    L.nt_collect.argtypes = [vp, ctypes.c_long, i32, f64, i32, u64, i64, P(ctypes.c_uint8), P(i64)]
    L.nt_stack.argtypes, L.nt_stack.restype = [P(vp), i32, P(i64)], vp
    L.nt_best_move.argtypes, L.nt_best_move.restype = [vp, u64, u64, i32, f64], i32
    L.nt_search_value.argtypes, L.nt_search_value.restype = [vp, u64, u64, i32, f64], f64
    L.nt_play.argtypes = [vp, ctypes.c_long, i32, f64, i32, u64, P(i64), P(i32), P(i32)]
    L.nt_play_from.argtypes = [vp, ctypes.c_long, ctypes.c_long, i32, f64, i32, u64, P(i64), P(i32), P(i32)]
    L.nt_play_trace.argtypes = [vp, ctypes.c_long, i32, f64, u64, ctypes.c_long, P(u64), P(ctypes.c_int8),
                                P(ctypes.c_long), P(i32), P(i32)]
    L.nt_play_trace.restype = i64
    L.nt_best_choose.argtypes, L.nt_best_choose.restype = [vp, u64, u64, i32, i32, P(i32), P(i32)], i32
    L.nt_choose_value.argtypes, L.nt_choose_value.restype = [vp, u64, u64, i32, i32], f64
    L.nt_play_choose.argtypes = [vp, ctypes.c_long, i32, i32, i32, u64, ctypes.c_long, ctypes.c_long,
                                 P(i64), P(i32), P(i32)]
    L.nt_train_choose.argtypes = [vp, ctypes.c_long, i32, f32, f32, u64, i32, i32, ctypes.c_long, f64,
                                  P(u64), ctypes.c_long, f64, P(i64), P(i32), P(i32)]
    L.nt_beam_choose.argtypes, L.nt_beam_choose.restype = [vp, u64, u64, i32, i32, i32, f64, i32, P(i32), P(i32)], i32
    L.nt_snake_score.argtypes, L.nt_snake_score.restype = [u64, u64, f64], f64
    L.nt_set_snake_decay.argtypes = [vp, f64]
    L.nt_snake_decay.argtypes, L.nt_snake_decay.restype = [vp], f64
    L.nt_set_leaf_snake.argtypes = [vp, f64]
    L.nt_downgrade.argtypes = [u64, u64, P(u64), P(u64)]
    L.nt_root_board.argtypes = [vp, u64, u64, P(u64), P(u64)]
    L.nt_set_downgrade.argtypes = [vp, i32]
    L.nt_set_endgame.argtypes = [vp, i32]
    L.nt_get_endgame.argtypes, L.nt_get_endgame.restype = [vp], i32
    L.nt_endgame_bits.argtypes = [i32]
    L.nt_endgame_full.restype = i32
    L.nt_endgame_used.argtypes, L.nt_endgame_used.restype = [i32], u64
    L.nt_endgame_value.argtypes, L.nt_endgame_value.restype = [u64, u64, i32], f32
    L.nt_endgame_load.argtypes, L.nt_endgame_load.restype = [ctypes.c_char_p, i32, P(i32), i32, i32], i32
    L.nt_endgame_loaded.restype = i32
    L.nt_endgame_load_path.argtypes = [ctypes.c_char_p, i32, P(i32), i32, i32, P(i32)]
    L.nt_endgame_load_path.restype = i32
    L.nt_endgame_loaded_family.argtypes, L.nt_endgame_loaded_family.restype = [i32], i32
    L.nt_endgame_families.restype = i32
    L.nt_endgame_value_family.argtypes, L.nt_endgame_value_family.restype = [u64, u64, i32, i32], f32
    L.nt_reach.argtypes = [vp, P(u64), ctypes.c_long, i32, ctypes.c_uint, i32, i32, u64, P(i32), P(i32)]
    L.nt_play_boards.argtypes = [vp, P(u64), ctypes.c_long, i32, i32, u64, P(i32), P(ctypes.c_int64), P(i32),
                                 P(ctypes.c_uint32)]
    L.nt_get_downgrade.argtypes, L.nt_get_downgrade.restype = [vp], i32
    L.nt_expected_target.argtypes, L.nt_expected_target.restype = [vp, u64, u64], f32
    L.nt_set_expect.argtypes = [vp, i32]
    L.nt_get_expect.argtypes, L.nt_get_expect.restype = [vp], i32
    L.nt_set_tt_bits.argtypes = [vp, i32]
    L.nt_get_tt_bits.argtypes, L.nt_get_tt_bits.restype = [vp], i32
    L.nt_set_depth_caps.argtypes = [vp, P(i32)]
    L.nt_get_depth_caps.argtypes = [vp, P(i32)]
    L.nt_leaf_snake.argtypes, L.nt_leaf_snake.restype = [vp], f64
    L.nt_beam_value.argtypes, L.nt_beam_value.restype = [vp, u64, u64, i32, i32, i32, f64, i32], f64
    L.nt_beam_plan.argtypes, L.nt_beam_plan.restype = [vp, u64, u64, i32, i32, i32, f64, i32, P(i32), P(i32), P(i32)], i32
    L.nt_play_beam.argtypes = [vp, ctypes.c_long, i32, i32, i32, i32, f64, i32, i32, u64, ctypes.c_long, ctypes.c_long,
                               P(i64), P(i32), P(i32)]
    L.nt_save_ex.argtypes, L.nt_save_ex.restype = [vp, cp, i32], i32
    L.nt_load_ex.argtypes, L.nt_load_ex.restype = [cp, P(i64), i32, i32], vp
    _lib = L
    return L


# --- board helpers --------------------------------------------------------------

def to_bits(board) -> int:
    """(4, 4) exponents (0 = empty, k = tile 2**k) -> bitboard: a Python int, five
    bits per cell. For a board of tile values, as in `Game.board`, use `tiles_to_bits`."""
    arr = np.asarray(board).reshape(16)
    if arr.max() > MAX_EXP:
        raise ValueError(f"exponents go up to {MAX_EXP}; for tile values use tiles_to_bits")
    bits = 0
    for i, v in enumerate(arr):
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


SNAKE_PATH = [0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12]
# 32768 16384 8192 on the top row, 4096 2048 1024 under them right to left (a 2x3 block in the corner)
BLOCK_PATH = [0, 1, 2, 6, 5, 4, 8, 9, 10, 11, 7, 3, 15, 14, 13, 12]


def endgame_value(bits: int, k: int, family: int = 0) -> float:
    """The endgame table's chance of winning for a board with k tiles locked on the
    family's path (0 = the snake) as it stands (not turned), or -1 if it is not such a position."""
    return float(lib().nt_endgame_value_family(*_lohi(bits), int(k), int(family)))


def load_endgame(directory, kmin: int | None = None, drops: list[int] | None = None, kcap: int | None = None,
                 path: list[int] | None = None) -> int:
    """Use endgame tables built in advance: <directory>/k<k>.u16 for k = kmin .. 15. kmin,
    the caps (drops), the largest capped table (kcap) and the formation's cell path are read
    from <directory>/caps.txt ("kmin d0,d1,... kcap 2 [path]"; no path = the snake) unless given.
    Each formation once per process; several formations can be loaded and play takes the best.
    Returns the lowest table loaded. Call before any play."""
    from pathlib import Path
    d = Path(directory)
    words = (d / "caps.txt").read_text().split() if None in (kmin, drops, kcap) else []
    if words and (len(words) < 4 or words[3] != "2"):
        raise ValueError(f"{d / 'caps.txt'} should read 'kmin d0,d1,... kcap 2' (layout 2, notes/tb/build2.c)")
    kmin = int(words[0]) if kmin is None else kmin
    drops = [int(x) for x in words[1].split(",") if x] if drops is None else drops
    kcap = (int(words[2]) if len(words) > 2 else 15) if kcap is None else kcap
    if path is None and len(words) > 4:
        path = [int(x) for x in words[4].split(",")]
    arr = (ctypes.c_int32 * 16)(*drops)
    p = (ctypes.c_int32 * 16)(*path) if path else None
    if lib().nt_endgame_load_path(str(d).encode(), int(kmin), arr, len(drops), int(kcap), p) < 0:
        raise ValueError(f"no tables for {kmin} locked and up in {d} with caps {drops} (or that formation is loaded)")
    return int(kmin)


def downgrade(bits: int) -> int:
    """Halve every tile above the largest tile value missing below the board's
    maximum (tile downgrading, Guei et al. 2022). Unchanged when nothing is missing."""
    lo, hi = ctypes.c_uint64(0), ctypes.c_uint64(0)
    lib().nt_downgrade(*_lohi(bits), ctypes.byref(lo), ctypes.byref(hi))
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


def pack_starts(starts):
    """Bitboards, or an (n, 4, 4) array of exponent grids as `save_starts` writes
    them -> (array of lo/hi halves, count), the form the native calls take. Pack a
    large set once and pass the result to `train(starts=...)` every chunk."""
    if isinstance(starts, np.ndarray):
        c = starts.reshape(-1, 16).astype(np.uint64)             # cell i sits in bits 5i..5i+4
        lohi = np.zeros((len(c), 2), dtype=np.uint64)
        for i in range(16):
            if 5 * i < 64:
                lohi[:, 0] |= c[:, i] << np.uint64(5 * i)        # cell 12 straddles: its low 4 bits land here
            if 5 * i + 5 > 64:
                lohi[:, 1] |= c[:, i] >> np.uint64(64 - 5 * i) if 5 * i < 64 else c[:, i] << np.uint64(5 * i - 64)
        flat = np.ascontiguousarray(lohi).reshape(-1)
        return (ctypes.c_uint64 * max(2, len(flat))).from_buffer_copy(flat.tobytes().ljust(16, b"\0")), len(c)
    starts = list(starts if starts is not None else [])
    return (ctypes.c_uint64 * max(2, 2 * len(starts)))(*[h for b in starts for h in _lohi(b)]), len(starts)


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


def snake_score(bits: int, decay: float = SNAKE_DECAY) -> float:
    """Tiles read along the snake path (0 1 2 3 / 7 6 5 4 / ...), the k-th weighted
    decay**k, best of the 8 symmetries. A chain laid out as a snake scores highest.
    A search uses its network's decay (`NTupleNet.snake_decay`)."""
    return float(lib().nt_snake_score(*_lohi(bits), float(decay)))


def stage_of(bits: int, boundaries: list[int]) -> int:
    """Stage index: how many boundaries the board's tile set has reached. The set is
    measured as the sum of the distinct tile values present, so the boundary 16384
    means "holds a 16384 or larger" and 49152 "holds a 32768 and a 16384, or larger"."""
    arr = (ctypes.c_int64 * max(1, len(boundaries)))(*boundaries)
    return int(lib().nt_stage_of(*_lohi(bits), arr, len(boundaries)))


def _game_buffers(games: int):
    """Per-game output arrays for the native play and train calls: scores, max tiles, moves."""
    return (ctypes.c_int64 * games)(), (ctypes.c_int32 * games)(), (ctypes.c_int32 * games)()


def _stats(scores, maxtiles, moves) -> dict:
    s, t, m = (np.ctypeslib.as_array(x).copy() for x in (scores, maxtiles, moves))
    return {"scores": s, "max_tiles": t, "moves": m}


# --- network --------------------------------------------------------------------

class NTupleNet:
    """`boundaries` are thresholds on which tiles the board holds (see `stage_of`) that
    switch to a fresh set of weight tables, e.g. [16384, 24576, 32768]. Entries a
    later stage has never updated read through to the previous stage.

    `tc_stages` is how many leading stages learn with temporal coherence (per-entry
    adaptive rates, 12 extra bytes per weight in memory); the rest use plain TD with their own
    step size (`alpha_plain`). Default: all stages when `tc`, else none.

    `v_init` is the value a fresh network gives every board (optimistic
    initialisation, Guei et al. 2022): each weight starts at v_init / (8 * patterns)
    and the greedy policy explores boards it has not played until their values
    come down. 320,000 is the published setting for eight 6-cell patterns."""

    def __init__(self, patterns=None, tc: bool = True, boundaries=None, tc_stages: int | None = None,
                 v_init: float = 0.0, _handle=None) -> None:
        self._lib = lib()
        if _handle is not None:
            self._h = _handle
        else:
            patterns = [list(p) for p in (patterns or DEFAULT_PATTERNS)]
            if not 1 <= len(patterns) <= MAX_PATTERNS or any(not 1 <= len(p) <= MAX_CELLS for p in patterns):
                raise ValueError("1..16 patterns of 1..7 cells each")
            if any(not 0 <= int(c) <= 15 for p in patterns for c in p):
                raise ValueError("pattern cells are board indices 0..15 (row * 4 + col)")
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
                raise MemoryError("could not allocate the n-tuple tables")
            if v_init:
                self._lib.nt_fill(self._h, float(v_init) / (8 * len(patterns)))

    @classmethod
    def load(cls, path, boundaries=None, keep_tc: bool = True, tc_stages: int | None = None) -> "NTupleNet":
        """With `boundaries`, a single-stage file becomes stage 0 of a multi-stage
        network (later stages start by reading through to it). A file that already
        has stages must be loaded with no boundaries or the same ones. keep_tc=False
        drops the temporal-coherence state everywhere; `tc_stages=n` keeps it on the
        first n stages only (plain TD elsewhere, at a quarter of the memory per stage)."""
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

    @classmethod
    def stack(cls, nets, boundaries) -> "NTupleNet":
        """A multi-stage network for play from single-stage networks trained one
        stage at a time: `nets[s]` becomes stage s, and nothing is read through from
        an earlier stage. Weights only. The inputs are copied and stay usable."""
        nets, bl = list(nets), [int(b) for b in boundaries]
        if len(bl) != len(nets) - 1 or bl != sorted(bl):
            raise ValueError("stack needs one ascending boundary between each pair of networks")
        handles = (ctypes.c_void_p * len(nets))(*[n._h for n in nets])
        h = lib().nt_stack(handles, len(nets), (ctypes.c_int64 * max(1, len(bl)))(*bl))
        if not h:
            raise ValueError("stack needs single-stage networks with the same patterns")
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
              alpha_plain: float | None = None, starts=None) -> dict:
        """Play `games` self-play games with TD(0) learning; returns per-game stats.
        `starts` (bitboards, or `pack_starts` of them) makes every game begin from
        one of those boards, drawn at random, with its score counted from there."""
        ap = alpha if alpha_plain is None else alpha_plain
        scores, maxtiles, moves = _game_buffers(games)
        packed, n = starts if isinstance(starts, tuple) else pack_starts(starts)
        self._lib.nt_train_from(self._h, games, threads, float(alpha), float(ap), seed & 0xFFFFFFFFFFFFFFFF,
                                packed, n, scores, maxtiles, moves)
        return _stats(scores, maxtiles, moves)

    def collect_starts(self, games: int, tile: int = 16384, depth: int = 0, cutoff: float = 0.0,
                       threads: int = 1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Play `games` games and keep from each the first board that holds `tile`
        or larger. `tile` is read as a stage boundary is (see `stage_of`): 16384 is
        a 16384 or larger, 49152 a 32768 and a 16384, or larger. Returns (grids,
        scores): (n, 4, 4) uint8 exponent grids for `save_starts`, and the points
        each game had at that board. Games that end before it are left out."""
        cells = np.zeros((games, 16), dtype=np.uint8)
        scores = (ctypes.c_int64 * games)()
        self._lib.nt_collect(self._h, games, int(depth), float(cutoff), threads, seed & 0xFFFFFFFFFFFFFFFF,
                             int(tile), cells.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), scores)
        reached = cells.any(axis=1)
        return cells[reached].reshape(-1, 4, 4), np.ctypeslib.as_array(scores).copy()[reached]

    @property
    def snake_decay(self) -> float:
        """Weight ratio between consecutive cells on the snake path, for every search
        on this network that adds a snake bonus (the beam's `snake`, `leaf_snake`).
        0.5 by default: the head dominates; nearer 1 the whole chain's order counts.
        Set it between searches, not while games are being played."""
        return float(self._lib.nt_snake_decay(self._h))

    @snake_decay.setter
    def snake_decay(self, decay: float) -> None:
        self._lib.nt_set_snake_decay(self._h, float(decay))

    @property
    def leaf_snake(self) -> float:
        """Weight of `snake_score` added to the leaves of the random-spawn search
        (best_move, search_value, play); 0 = the tables alone."""
        return float(self._lib.nt_leaf_snake(self._h))

    @leaf_snake.setter
    def leaf_snake(self, weight: float) -> None:
        self._lib.nt_set_leaf_snake(self._h, float(weight))

    @property
    def downgrade(self) -> int:
        """Tile (e.g. 32768) from which the random-spawn search picks its move on the
        downgraded board (see `downgrade`); 0 = never. Boards with the largest tiles
        are the least trained, and their downgraded twins are not."""
        exp = int(self._lib.nt_get_downgrade(self._h))
        return (1 << exp) if exp else 0

    @downgrade.setter
    def downgrade(self, tile: int) -> None:
        self._lib.nt_set_downgrade(self._h, int(tile).bit_length() - 1 if tile else 0)

    @property
    def endgame(self) -> int:
        """Exact endgame tables choose the move once this many tiles, 32768 down, lie
        locked along a snake (in any orientation); 0 = off. The tables are exact for
        that formation and shared by every network in the process; they fill as
        positions are reached and take gigabytes for 5 or 6 locked tiles."""
        return int(self._lib.nt_get_endgame(self._h))

    @endgame.setter
    def endgame(self, k: int) -> None:
        self._lib.nt_set_endgame(self._h, int(k))

    @property
    def expect_target(self) -> bool:
        """Train toward the expected-value target: the average, over every tile that can
        appear on the previous afterstate, of the best move's reward and value, instead
        of the one tile that did appear (Wang and Matsuzaki). The same answer in the
        long run with less noise per update, at 10 to 15 times the cost per move.
        Single-stage networks; `train` only. Set it between calls."""
        return bool(self._lib.nt_get_expect(self._h))

    @expect_target.setter
    def expect_target(self, on: bool) -> None:
        self._lib.nt_set_expect(self._h, int(bool(on)))

    def expected_target(self, bits: int) -> float:
        """The expected-value target of an afterstate under the current weights."""
        return float(self._lib.nt_expected_target(self._h, *_lohi(bits)))

    @property
    def tt_bits(self) -> int:
        """Size of the search's transposition table: 2**tt_bits entries of 32 bytes
        for each search thread (20 by default, 32 MB), beside a small fixed table for
        the search's leaves. Deep searches want more.
        It changes the speed only, never the move chosen. Set it between searches."""
        return int(self._lib.nt_get_tt_bits(self._h))

    @tt_bits.setter
    def tt_bits(self, bits: int) -> None:
        self._lib.nt_set_tt_bits(self._h, int(bits))

    @property
    def depth_caps(self) -> list[int]:
        """Random-spawn search: the deepest search allowed for a board with i empty
        cells, for i = 0..16 (0 = no cap). `[5, 5, 5, 5, 4, 4, 4, 4, 3]` searches
        crowded boards at up to depth 5 and boards with 8 or more empty cells at 3;
        a list shorter than 17 is extended with its last value. Set it between searches."""
        out = (ctypes.c_int32 * 17)()
        self._lib.nt_get_depth_caps(self._h, out)
        return list(out)

    @depth_caps.setter
    def depth_caps(self, caps) -> None:
        caps = [int(c) for c in (caps or [])]
        if caps:
            caps = (caps + [caps[-1]] * 17)[:17]
        self._lib.nt_set_depth_caps(self._h, (ctypes.c_int32 * 17)(*caps) if caps else None)

    def root_board(self, bits: int) -> int:
        """The board the search evaluates for this root (downgraded while it holds the threshold tile)."""
        lo, hi = ctypes.c_uint64(0), ctypes.c_uint64(0)
        self._lib.nt_root_board(self._h, *_lohi(bits), ctypes.byref(lo), ctypes.byref(hi))
        return lo.value | (hi.value << 64)

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
        scores, maxtiles, moves = _game_buffers(games)
        self._lib.nt_play_choose(self._h, games, int(depth), int(topk), threads, seed & 0xFFFFFFFFFFFFFFFF,
                                 int(max_moves), int(prefix), scores, maxtiles, moves)
        return _stats(scores, maxtiles, moves)

    # --- beam search: plan a line of the deterministic choose game ------------

    def beam_value(self, bits: int, width: int = 64, depth: int = 16, spread: int = 0, snake: float = 0.0,
                   tiles: int = 3) -> float:
        """Value of the best line found by a beam of `width` boards over `depth`
        move+placement steps; unlimited width equals the full tree of that depth.
        `spread` > 0 caps the survivors per parent so the beam covers distinct lines;
        `snake` adds that weight times `snake_score` to every leaf. `tiles` is a mask of
        the placements allowed: 1 = 2s only, 2 = 4s only, 3 = both; adding 4 charges each
        placed 4 the 4 points it forfeits (ranking only). Adding 8 plans with 2s first and
        allows 4s only when the best 2s-only line dies within `depth` steps; such a fallback
        plan is one step long, so the next plan tries 2s again (8 or 12 play a whole game
        with only the 4s that survival, e.g. the final cascade to 131072, needs). Adding
        16 to that makes the fallback sparing: it looks for a line that survives with one
        4, then two, and so on, and a tile that ends the game whatever it is goes down as
        a 4 (28 = 2s first, 4s charged, as few as possible)."""
        return float(self._lib.nt_beam_value(self._h, *_lohi(bits), int(width), int(depth), int(spread), float(snake),
                                             int(tiles)))

    def beam_choose(self, bits: int, width: int = 64, depth: int = 16, spread: int = 0,
                    snake: float = 0.0, tiles: int = 3) -> tuple[int, int, int]:
        """(move, cell 0..15, tile exponent 1 or 2) starting the best line; move -1 if stuck."""
        cell_, value = ctypes.c_int(-1), ctypes.c_int(0)
        m = int(self._lib.nt_beam_choose(self._h, *_lohi(bits), int(width), int(depth), int(spread), float(snake),
                                         int(tiles), ctypes.byref(cell_), ctypes.byref(value)))
        return m, cell_.value, value.value

    def beam_plan(self, bits: int, width: int = 64, depth: int = 16, spread: int = 0,
                  snake: float = 0.0, tiles: int = 3) -> list[tuple[int, int, int]]:
        """The best line found: [(move, cell, tile exponent), ...], empty when stuck."""
        depth = min(max(int(depth), 1), BEAM_MAX_DEPTH)   # the C side clamps the same way and fills `depth` slots
        moves, cells, values = (ctypes.c_int * depth)(), (ctypes.c_int * depth)(), (ctypes.c_int * depth)()
        n = int(self._lib.nt_beam_plan(self._h, *_lohi(bits), int(width), depth, int(spread), float(snake),
                                       int(tiles), moves, cells, values))
        return [(moves[i], cells[i], values[i]) for i in range(n)]

    def play_beam(self, games: int, width: int = 64, depth: int = 16, threads: int = 1, seed: int = 0,
                  max_moves: int = 200_000, stride: int = 1, spread: int = 0, prefix: int = 0,
                  snake: float = 0.0, tiles: int = 3) -> dict:
        """Play with beam search; `stride` steps of each plan are committed before
        re-planning; the first `prefix` moves get random spawns (see play_choose)."""
        scores, maxtiles, moves = _game_buffers(games)
        self._lib.nt_play_beam(self._h, games, int(width), int(depth), int(stride), int(spread), float(snake),
                               int(tiles), threads, seed & 0xFFFFFFFFFFFFFFFF, int(max_moves), int(prefix),
                               scores, maxtiles, moves)
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
        scores, maxtiles, moves = _game_buffers(games)
        starts = list(starts or [])
        packed = (ctypes.c_uint64 * max(2, 2 * len(starts)))(*[h for b in starts for h in _lohi(b)])
        self._lib.nt_train_choose(self._h, games, threads, float(alpha), float(ap), seed & 0xFFFFFFFFFFFFFFFF,
                                  int(depth), int(topk), int(max_moves), float(explore),
                                  packed, len(starts), float(start_frac), scores, maxtiles, moves)
        return _stats(scores, maxtiles, moves)

    def play(self, games: int, depth: int = 1, cutoff: float = 0.0, threads: int = 1, seed: int = 0,
             first: int = 0) -> dict:
        """Play games `first .. first + games - 1` of `seed` with expectimax at `depth`.
        A game depends only on the seed and its number (with cutoff 0), not on the
        threads or on which other games are played with it."""
        scores, maxtiles, moves = _game_buffers(games)
        self._lib.nt_play_from(self._h, games, int(first), int(depth), float(cutoff), threads,
                               seed & 0xFFFFFFFFFFFFFFFF, scores, maxtiles, moves)
        return _stats(scores, maxtiles, moves)

    def reach(self, starts, k: int = 6, depth: int = 0, threads: int = 1, seed: int = 0, families: int = 1) -> dict:
        """From each start board (bitboards or an (n, 4, 4) grid array), play until the board
        is an endgame-table position with k or more locked tiles or the game is lost, with
        this network's search settings (tables off). `families` is a bit mask of the formations
        that count: 1 = the snake, 2 = the next formation loaded with `load_endgame`, and so on
        (formations other than the snake count only where their tables are loaded).
        Returns `reached` (0/1) and `moves`."""
        packed, n = starts if isinstance(starts, tuple) else pack_starts(starts)
        reached, moves = (ctypes.c_int32 * max(1, n))(), (ctypes.c_int32 * max(1, n))()
        self._lib.nt_reach(self._h, packed, n, int(k), int(families), int(depth), threads, seed & 0xFFFFFFFFFFFFFFFF,
                           reached, moves)
        return {"reached": np.ctypeslib.as_array(reached)[:n].copy(), "moves": np.ctypeslib.as_array(moves)[:n].copy()}

    def play_boards(self, starts, depth: int = 0, threads: int = 1, seed: int = 0) -> dict:
        """From each start board (bitboards or an (n, 4, 4) grid array), play to the end with
        this network's settings (search depth, downgrading, endgame tables). Returns per game
        `max_tile`, `score` (points from the start board on), `moves`, and `seen`: bit e set
        when a tile 2^e was on the board at some point (for milestones such as a 32768 made
        beside a 65536)."""
        packed, n = starts if isinstance(starts, tuple) else pack_starts(starts)
        tile, moves = (ctypes.c_int32 * max(1, n))(), (ctypes.c_int32 * max(1, n))()
        score = (ctypes.c_int64 * max(1, n))()
        seen = (ctypes.c_uint32 * max(1, n))()
        self._lib.nt_play_boards(self._h, packed, n, int(depth), threads, seed & 0xFFFFFFFFFFFFFFFF, tile, score, moves,
                                 seen)
        return {k: np.ctypeslib.as_array(v)[:n].copy()
                for k, v in (("max_tile", tile), ("score", score), ("moves", moves), ("seen", seen))}

    def play_trace(self, game: int, depth: int = 1, cutoff: float = 0.0, seed: int = 0,
                   max_boards: int = 200_000) -> dict:
        """Game number `game` of `seed`, exactly as `play` plays it, with its history:
        `boards[t]` is the board before move t (an (n, 4, 4) array of exponents) and
        `dirs[t]` the move made there (0 up, 1 right, 2 down, 3 left); the last board
        is the final one, with dir -1. Also `score`, `max_tile`, `moves`."""
        buf = (ctypes.c_uint64 * (2 * max_boards))()
        dirs = (ctypes.c_int8 * max_boards)()
        n, tile, moves = ctypes.c_long(0), ctypes.c_int(0), ctypes.c_int(0)
        score = self._lib.nt_play_trace(self._h, int(game), int(depth), float(cutoff), seed & 0xFFFFFFFFFFFFFFFF,
                                        max_boards, buf, dirs, ctypes.byref(n), ctypes.byref(tile), ctypes.byref(moves))
        raw = np.ctypeslib.as_array(buf)[: 2 * n.value].reshape(-1, 2)
        cells = np.empty((n.value, 16), dtype=np.uint8)
        for i in range(16):                          # 5 bits a cell, 12 cells in the low half and 4 across/above
            lo_bits, shift = CELL_BITS * i, CELL_BITS * i - 64
            if lo_bits + CELL_BITS <= 64:
                cells[:, i] = (raw[:, 0] >> np.uint64(lo_bits)) & np.uint64(CELL_MASK)
            elif shift >= 0:
                cells[:, i] = (raw[:, 1] >> np.uint64(shift)) & np.uint64(CELL_MASK)
            else:                                    # the cell that straddles the two halves
                cells[:, i] = ((raw[:, 0] >> np.uint64(lo_bits)) | (raw[:, 1] << np.uint64(-shift))) & np.uint64(CELL_MASK)
        return {"score": int(score), "max_tile": tile.value, "moves": moves.value,
                "boards": cells.reshape(-1, 4, 4), "dirs": np.ctypeslib.as_array(dirs)[: n.value].copy()}


def percent(rate: float, digits: int = 0) -> str:
    """A rate as a percentage, rounded down: a report never claims a level not yet reached."""
    scale = 10 ** digits
    return f"{math.floor(rate * 100 * scale + 1e-9) / scale:.{digits}f}%"


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
        **{f"rate_{v}": float((t >= v).mean()) for v in (1024, 2048, 4096, 8192, 16384, 32768, 65536)},
    }


class NTupleAgent:
    """Plays with a trained n-tuple network plus expectimax. Registered as "ntuple";
    "ntuple-cool" is the same class on tables trained for cool mode."""

    def __init__(self, weights=None, depth: int = 3, cutoff: float = 0.0, choose_depth: int = 3,
                 topk: int = 4, beam_width: int = 0, beam_depth: int = 16, beam_stride: int = 1,
                 beam_spread: int = 0, beam_snake: float = 0.0, snake_decay: float | None = None,
                 beam_tiles: int = 3, leaf_snake: float = 0.0, net: "NTupleNet | None" = None) -> None:
        if net is not None:                 # a ready network, e.g. an empty one for a heuristic-only player
            self.net = net
        else:
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
        self.beam_tiles = beam_tiles      # placement mask, see NTupleNet.beam_value (28 = 2s first, as few 4s as possible)
        if snake_decay is not None:       # how fast the snake weights fall off along the path
            self.net.snake_decay = snake_decay
        if leaf_snake:                    # snake bonus at the random-spawn search leaves
            self.net.leaf_snake = leaf_snake
        self._plan: list[tuple[int, int, int]] = []
        self._plan_board = -1             # board the next cached step applies to

    def _beam_step(self, bits: int) -> tuple[int, int, int]:
        if not self._plan or self._plan_board != bits:
            self._plan = self.net.beam_plan(bits, self.beam_width, self.beam_depth, self.beam_spread,
                                            self.beam_snake, self.beam_tiles)[: max(1, self.beam_stride)]
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
