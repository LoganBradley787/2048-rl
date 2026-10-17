"""The independent rules checker (examples/rules2048.py) against the engine's games and against broken traces."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import rules2048 as R  # noqa: E402

from game2048 import ntuple as nt  # noqa: E402


def test_slides_and_merges_by_the_rules():
    assert R.slide_row_left([1, 1, 1, 1]) == ([2, 2, 0, 0], 8)
    assert R.slide_row_left([2, 1, 1, 0]) == ([2, 2, 0, 0], 4)        # a new 4 does not merge with the 4 that was there
    assert R.slide_row_left([0, 3, 0, 3]) == ([4, 0, 0, 0], 16)
    assert R.slide_row_left([1, 2, 1, 2]) == ([1, 2, 1, 2], 0)
    b = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 2]]
    assert R.apply_move(b, 0)[0] == [[1, 0, 1, 2], [0] * 4, [0] * 4, [0] * 4]
    assert R.apply_move(b, 1)[0] == [[0, 0, 0, 1], [0] * 4, [0, 0, 0, 1], [0, 0, 0, 2]]


def test_agrees_with_the_engine_on_random_boards():
    rng = np.random.default_rng(5)
    for _ in range(2000):
        g = rng.integers(0, 12, size=(4, 4)) * (rng.random((4, 4)) < 0.7)
        bits = nt.to_bits(g.astype(np.uint8))
        for d in range(4):
            after, pts = nt.move(bits, d)
            mine, p = R.apply_move(g.tolist(), d)
            assert nt.from_bits(after).tolist() == mine and pts == p


def test_accepts_engine_games_and_rejects_broken_ones():
    net = nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False)
    s = net.play_trace(3, depth=0, seed=11, max_boards=100_000)
    boards, dirs, score = s["boards"], s["dirs"], int(s["score"])
    assert R.check_trace(boards, dirs, score) is None
    assert R.check_trace(boards, dirs, score + 4) is not None
    bad = boards.copy()
    bad[len(bad) // 2][0][0] += 1
    assert R.check_trace(bad, dirs, score) is not None
    wrong = dirs.copy()
    wrong[5] = (wrong[5] + 1) % 4
    assert R.check_trace(boards, wrong, score) is not None
    assert R.check_trace(boards[:-1], np.append(dirs[:-2], -1), None) is not None   # stopped with moves left
    net.close()
