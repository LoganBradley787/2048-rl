"""Convert weights released with TDL2048 (https://github.com/moporgic/TDL2048,
Guei et al.) to this engine's format, to check the search here against the
numbers published for them.

    uv run python examples/import_tdl2048.py 8x6patt.w.xz --out checkpoints/tdl2048/weights.bin

A TDL2048 file is a list of tables named by their pattern (hex cell numbers,
"012456"); a pattern that appears again starts the next stage, so a two-stage
file needs --tiles 16384 for the boundary. Their index puts a pattern's first
cell in the lowest four bits; the tables here are radix 18 with the first cell
most significant.
"""

from __future__ import annotations

import argparse
import lzma
import struct
from pathlib import Path

import numpy as np

from game2048 import ntuple as nt


def read_tables(path) -> list[tuple[str, np.ndarray]]:
    """[(pattern name, float32 table)] in file order."""
    f = lzma.open(path, "rb") if str(path).endswith(".xz") else open(path, "rb")
    with f:
        if f.read(1) != b"\x00":
            raise ValueError("not a TDL2048 weights file")
        tables = []
        for _ in range(struct.unpack("<I", f.read(4))[0]):
            if f.read(1) != b"\x04":
                raise ValueError("unknown table format")
            raw = f.read(8)
            sign, width, zero = struct.unpack("<IHH", raw)
            unit, n = struct.unpack("<HQ", f.read(10))
            name = f"{sign:0{width or (n.bit_length() - 1) // 4}x}" if zero == 0 else raw.decode().strip()
            dtype = {2: "<f2", 4: "<f4", 8: "<f8"}[unit]
            tables.append((name, np.frombuffer(f.read(unit * n), dtype=dtype).astype(np.float32)))
            while (unit := struct.unpack("<H", f.read(2))[0]):   # learning state and anything else: skipped
                f.read(unit * struct.unpack("<Q", f.read(8))[0])
        return tables


def to_radix18(table: np.ndarray, n: int) -> np.ndarray:
    """A 16**n table indexed by sum(cell_j << 4j) -> 18**n indexed with cell 0 most significant."""
    old = np.arange(16 ** n, dtype=np.int64)
    new = np.zeros_like(old)
    for j in range(n):
        new += ((old >> (4 * j)) & 15) * 18 ** (n - 1 - j)
    out = np.zeros(18 ** n, dtype=np.float32)
    out[new] = table
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", help="TDL2048 weights, .w or .w.xz")
    p.add_argument("--out", required=True)
    p.add_argument("--tiles", default="", help="comma-separated stage boundaries for a multi-stage file, e.g. 16384")
    args = p.parse_args(argv)

    stages: list[dict[str, np.ndarray]] = [{}]
    for name, table in read_tables(args.file):
        if name in stages[-1]:
            stages.append({})
        stages[-1][name] = table
    names = list(stages[0])
    bounds = [int(x) for x in args.tiles.split(",") if x.strip()]
    if len(bounds) != len(stages) - 1 or any(list(s) != names for s in stages):
        raise SystemExit(f"{len(stages)} stages of patterns {[list(s) for s in stages]}: give {len(stages) - 1} --tiles")
    patterns = [[int(ch, 16) for ch in name] for name in names]
    if any(len(t) != 16 ** len(pt) for s in stages for t, pt in zip(s.values(), patterns)):
        raise SystemExit("a table's size does not match its pattern")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    nets = []
    for s in stages:                                   # one single-stage file per stage, then stacked
        tmp = Path(args.out).with_suffix(".stage.tmp")
        with open(tmp, "wb") as f:                     # format 06, weights only: see nt_save_ex
            f.write(b"NTUPLE06" + struct.pack("<iiii", len(patterns), 0, 0, 0) + struct.pack("<8q", *([0] * 8)))
            f.write(struct.pack(f"<{nt.MAX_PATTERNS}i", *[len(pt) for pt in patterns], *([0] * (nt.MAX_PATTERNS - len(patterns)))))
            cells = np.zeros((nt.MAX_PATTERNS, nt.MAX_LEN), dtype="<i4")
            for k, pt in enumerate(patterns):
                cells[k, : len(pt)] = pt
            f.write(cells.tobytes())
            for name, pt in zip(names, patterns):
                f.write(to_radix18(s[name], len(pt)).tobytes())
        nets.append(nt.NTupleNet.load(tmp))
        tmp.unlink()
    net = nets[0] if len(nets) == 1 else nt.NTupleNet.stack(nets, bounds)
    net.save(args.out, weights_only=True)
    print(f"wrote {args.out}: patterns {' '.join(names)}, {len(stages)} stage(s), boundaries {net.boundaries}")


if __name__ == "__main__":
    main()
