"""Average the weights of several checkpoints of one run into a single file.

    uv run python examples/average_weights.py --weights checkpoints/ntuple/snapshots/*.bin --out checkpoints/ntuple_avg/weights.bin

A network trained at a fixed learning rate keeps moving around the values it
has learned; the mean of snapshots taken along the way sits closer to them
than any one snapshot does (Guei's dissertation, after stochastic weight
averaging). The files must be weights-only saves of the same network shape:
`weights.bin` as the trainer writes it, copied aside from time to time.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

from game2048 import ntuple as nt

HEADER = 8 + 4 * 4 + 8 * nt.MAX_STAGES + 4 * nt.MAX_PATTERNS + 4 * nt.MAX_PATTERNS * nt.MAX_LEN   # format 06
CHUNK = 1 << 22                                                                                    # floats per step


def header(path: Path) -> bytes:
    with open(path, "rb") as f:
        h = f.read(HEADER)
    if h[:8] != b"NTUPLE06":
        raise SystemExit(f"{path}: not a current weights file (load and save it again to convert)")
    _, tc_stages, flags, _ = struct.unpack("<4i", h[8:24])
    if tc_stages or flags:
        raise SystemExit(f"{path}: holds learning state; average the weights-only file (weights.bin)")
    return h


def average(paths: list[Path], out: Path) -> None:
    head = header(paths[0])
    size = paths[0].stat().st_size
    for p in paths[1:]:
        if header(p) != head or p.stat().st_size != size:
            raise SystemExit(f"{p}: a different network shape from {paths[0]}")
    maps = [np.memmap(p, dtype="<f4", mode="r", offset=HEADER) for p in paths]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(head)
        for i in range(0, len(maps[0]), CHUNK):
            acc = np.zeros(min(CHUNK, len(maps[0]) - i), dtype=np.float64)
            for m in maps:
                acc += m[i:i + CHUNK]
            f.write((acc / len(maps)).astype("<f4").tobytes())
    tmp.replace(out)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", nargs="+", required=True, help="two or more weights-only files of the same network")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    paths = [Path(w) for w in args.weights]
    if len(paths) < 2:
        p.error("give at least two files")
    average(paths, Path(args.out))
    net = nt.NTupleNet.load(args.out)                  # a file the engine reads back, or this raises
    print(f"wrote {args.out}: the mean of {len(paths)} files, {len(net.patterns)} patterns, boundaries {net.boundaries}")
    net.close()


if __name__ == "__main__":
    main()
