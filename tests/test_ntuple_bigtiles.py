"""Tiles above 32768: five-bit cells, radix-18 tables, and conversion of old files."""
import json
from pathlib import Path

import numpy as np
import pytest

from game2048 import ntuple as nt

FIX = Path(__file__).parent / "fixtures"


def row_board(*rows):
    return nt.to_bits(np.array(rows, dtype=np.uint8))


def test_cells_hold_five_bits_and_round_trip():
    assert nt.CELL_BITS == 5
    grid = np.array([[17, 16, 15, 1], [0, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.uint8)
    bits = nt.to_bits(grid)
    assert nt.cell(bits, 0) == 17 and nt.cell(bits, 1) == 16 and nt.cell(bits, 15) == 12
    assert np.array_equal(nt.from_bits(bits), grid)
    assert nt.cell(nt.with_cell(0, 5, 17), 5) == 17
    assert nt.tiles_to_bits([[131072, 65536, 0, 0], [0] * 4, [0] * 4, [0] * 4]) == nt.with_cell(nt.with_cell(0, 0, 17), 1, 16)


def test_two_32768_tiles_merge_into_65536_and_then_131072():
    b = row_board([15, 15, 0, 0], [0] * 4, [0] * 4, [0] * 4)
    after, reward = nt.move(b, 3)                      # left
    assert nt.from_bits(after)[0].tolist() == [16, 0, 0, 0] and reward == 65536
    b = row_board([16, 16, 0, 0], [0] * 4, [0] * 4, [0] * 4)
    after, reward = nt.move(b, 3)
    assert nt.from_bits(after)[0].tolist() == [17, 0, 0, 0] and reward == 131072
    b = row_board([0, 0, 16, 16], [0] * 4, [0] * 4, [0] * 4)   # and vertically / rightwards
    after, reward = nt.move(b, 1)
    assert nt.from_bits(after)[0].tolist() == [0, 0, 0, 17] and reward == 131072
    b = row_board([16, 0, 0, 0], [16, 0, 0, 0], [0] * 4, [0] * 4)
    after, reward = nt.move(b, 0)
    assert nt.from_bits(after)[:, 0].tolist() == [17, 0, 0, 0] and reward == 131072


def test_tables_distinguish_32768_from_65536():
    n = nt.NTupleNet(patterns=[[0, 1, 2, 3]])
    grid = np.arange(2, 18, dtype=np.uint8).reshape(4, 4)     # 16 distinct exponents, 2..17
    a = nt.to_bits(grid)
    n.update(a, 80.0, alpha=1.0)
    assert n.value(a) == pytest.approx(80.0)
    # cell 0 (exponent 2) is read by 4 of the 8 symmetric samples of a row pattern;
    # changing it to 16 or 17 must miss exactly those, so the value halves
    assert n.value(nt.with_cell(a, 0, 16)) == pytest.approx(40.0)
    assert n.value(nt.with_cell(a, 0, 17)) == pytest.approx(40.0)
    assert n.value(nt.with_cell(nt.with_cell(a, 0, 16), 15, 15)) == pytest.approx(0.0)  # cells 0 and 15 cover every sample
    n.close()


@pytest.mark.parametrize("name", ["ntuple05_w.bin", "ntuple05_tc.bin"])
def test_old_radix16_files_load_with_the_same_values(name):
    fix = json.loads((FIX / "ntuple05.json").read_text())
    n = nt.NTupleNet.load(FIX / name)
    assert n.patterns == fix["patterns"]
    for grid, expected in zip(fix["boards"], fix["values"]):
        assert n.value(nt.to_bits(np.array(grid, dtype=np.uint8))) == pytest.approx(expected, rel=1e-6, abs=1e-6)
    assert n.tc == name.endswith("_tc.bin")
    stats = n.train(games=5, threads=1, seed=1)       # the converted learning state still trains
    assert (stats["scores"] > 0).all()
    n.close()


def test_saved_files_round_trip_big_tiles(tmp_path):
    n = nt.NTupleNet(patterns=[[0, 1, 2]])
    a = nt.to_bits(np.arange(2, 18, dtype=np.uint8).reshape(4, 4))   # includes 65536 and 131072
    n.update(a, 5.0, alpha=1.0)
    n.save(tmp_path / "w.bin")
    m = nt.NTupleNet.load(tmp_path / "w.bin")
    assert m.value(a) == pytest.approx(5.0)
    assert (tmp_path / "w.bin").read_bytes()[:8] == b"NTUPLE06"
    n.close(); m.close()
