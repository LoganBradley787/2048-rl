import numpy as np
import pytest

from game2048 import vec_env as ve
from game2048.core import Direction, Game, slide_and_merge_row


def rand_boards(rng, n, max_exp=11, density=0.6):
    b = rng.integers(1, max_exp + 1, size=(n, 4, 4)).astype(np.uint8)
    b[rng.random((n, 4, 4)) > density] = 0
    return b


def to_values(b):
    return [[int(2 ** int(v)) if v else 0 for v in row] for row in b]


# --- row tables -------------------------------------------------------------

def test_row_table_matches_core_merge_rule():
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 12, size=(2000, 4)).astype(np.uint8)
    idx = ve.encode_rows(rows[:, None, :])[:, 0]
    out = ve.decode_rows(ve.ROW_LUT[idx][:, None])[:, 0]
    reward = ve.REWARD_LUT[idx]
    for r, o, rw in zip(rows, out, reward):
        vals = [int(2 ** int(v)) if v else 0 for v in r]
        exp_vals, exp_rw = slide_and_merge_row(vals)
        assert [int(2 ** int(v)) if v else 0 for v in o] == exp_vals
        assert int(rw) == exp_rw


def test_encode_decode_round_trip():
    rng = np.random.default_rng(1)
    b = rand_boards(rng, 50, max_exp=15, density=1.0)
    assert np.array_equal(ve.decode_rows(ve.encode_rows(b)), b)


def test_two_32768_tiles_do_not_merge():
    # 2^16 cannot be encoded in a nibble; documented cap, never reached in practice.
    row = np.array([[[15, 15, 0, 0]]], dtype=np.uint8)
    out = ve.decode_rows(ve.ROW_LUT[ve.encode_rows(row)])
    assert out.tolist() == [[[15, 15, 0, 0]]]


# --- afterstates -------------------------------------------------------------

def test_afterstates_match_game_for_every_direction():
    rng = np.random.default_rng(2)
    boards = rand_boards(rng, 300)
    after, rewards, valid = ve.afterstates(boards)
    assert after.shape == (300, 4, 4, 4) and rewards.shape == (300, 4) and valid.shape == (300, 4)
    for i, b in enumerate(boards):
        vals = to_values(b)
        for d in Direction:
            exp_board, exp_reward = Game._apply(vals, d)
            assert to_values(after[i, d]) == exp_board, (i, d)
            assert int(rewards[i, d]) == exp_reward
            assert bool(valid[i, d]) == (exp_board != vals)


def test_afterstates_do_not_mutate_input():
    rng = np.random.default_rng(3)
    boards = rand_boards(rng, 10)
    copy = boards.copy()
    ve.afterstates(boards)
    assert np.array_equal(boards, copy)


# --- spawn -------------------------------------------------------------------

def test_spawn_adds_exactly_one_tile_where_there_is_room():
    rng = np.random.default_rng(4)
    boards = rand_boards(rng, 200, density=0.5)
    boards[0][:] = 1  # full board: nothing should change
    before = boards.copy()
    ve.spawn(boards, rng)
    assert np.array_equal(boards[0], before[0])
    for b, a in zip(before[1:], boards[1:]):
        diff = (b == 0) & (a != 0)
        assert diff.sum() == 1
        assert a[diff][0] in (1, 2)
        assert np.array_equal(b[~diff], a[~diff])


def test_spawn_is_a_two_about_ninety_percent_of_the_time():
    rng = np.random.default_rng(5)
    boards = np.zeros((20000, 4, 4), dtype=np.uint8)
    ve.spawn(boards, rng)
    fours = (boards == 2).sum()
    assert 0.08 < fours / len(boards) < 0.12


# --- VecEnv ------------------------------------------------------------------

def test_vec_env_reset_gives_two_tiles_and_zero_score():
    env = ve.VecEnv(8, seed=0)
    assert env.boards.shape == (8, 4, 4)
    assert all((b != 0).sum() == 2 for b in env.boards)
    assert env.scores.tolist() == [0] * 8


def test_vec_env_step_applies_move_spawns_and_scores():
    env = ve.VecEnv(2, seed=0)
    env.boards[0] = np.array([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)
    env.boards[1] = np.array([[1, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)
    rewards, done = env.step(np.array([3, 3]))  # LEFT, LEFT
    assert rewards.tolist() == [4, 0]
    assert done.tolist() == [False, False]
    assert env.boards[0, 0, 0] == 2 and (env.boards[0] != 0).sum() == 2
    assert env.boards[1].tolist() == [[1, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]  # invalid: untouched
    assert env.scores.tolist() == [4, 0]


def test_vec_env_reports_done_when_no_moves_remain():
    env = ve.VecEnv(1, seed=0)
    env.boards[0] = np.array([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 3]], dtype=np.uint8)
    # only merge: nothing... make the last two mergeable then fill via spawn
    env.boards[0, 3, 3] = 1  # row 3: 2,1,2,1 -> checkerboard, stuck
    assert env.legal_mask().tolist() == [[False, False, False, False]]
    assert env.done().tolist() == [True]


def test_vec_env_reset_mask_only_touches_selected_boards():
    env = ve.VecEnv(4, seed=0)
    env.scores[:] = 10
    before = env.boards.copy()
    env.reset(np.array([True, False, False, True]))
    assert env.scores.tolist() == [0, 10, 10, 0]
    assert np.array_equal(env.boards[1], before[1]) and np.array_equal(env.boards[2], before[2])
    assert (env.boards[0] != 0).sum() == 2 and (env.boards[3] != 0).sum() == 2


def test_greedy_rollout_on_vec_env_finishes():
    env = ve.VecEnv(16, seed=1)
    rng = np.random.default_rng(1)
    for _ in range(5000):
        mask = env.legal_mask()
        if not mask.any():
            break
        # random legal move for alive envs, action 0 for dead ones (ignored)
        scores = rng.random(mask.shape) * mask
        actions = scores.argmax(1)
        env.step(actions)
    assert env.done().all()
    assert (env.scores > 0).all()


# --- symmetries ---------------------------------------------------------------

def test_eight_symmetries_are_distinct_and_include_identity():
    rng = np.random.default_rng(6)
    b = rand_boards(rng, 1, density=1.0)
    outs = [ve.apply_symmetry(b, k)[0].tobytes() for k in range(8)]
    assert len(set(outs)) == 8
    assert np.array_equal(ve.apply_symmetry(b, 0), b)


def test_symmetry_preserves_best_reward():
    rng = np.random.default_rng(7)
    boards = rand_boards(rng, 100)
    _, r0, v0 = ve.afterstates(boards)
    best0 = np.where(v0, r0, -1).max(1)
    for k in range(8):
        _, rk, vk = ve.afterstates(ve.apply_symmetry(boards, k))
        assert np.array_equal(np.where(vk, rk, -1).max(1), best0)


def test_inverse_symmetry_round_trips():
    rng = np.random.default_rng(8)
    b = rand_boards(rng, 20, density=1.0)
    for k in range(8):
        assert np.array_equal(ve.apply_inverse_symmetry(ve.apply_symmetry(b, k), k), b)


# --- kind spawns ------------------------------------------------------------------

def test_kind_spawn_never_ends_a_board_when_it_can_avoid_it():
    rng = np.random.default_rng(9)
    # only (3,3) empty; a 4 there is fatal, a 2 merges with the 2 above
    b = np.array([[1, 2, 3, 4], [4, 3, 2, 5], [1, 2, 3, 1], [2, 1, 4, 0]], dtype=np.uint8)
    boards = np.repeat(b[None], 50, axis=0)
    ve.spawn(boards, rng, kind=True)
    assert (boards[:, 3, 3] == 1).all()
    assert ve.afterstates(boards)[2].any(axis=1).all()


def test_kind_spawn_falls_back_to_random_when_every_tile_is_fatal():
    rng = np.random.default_rng(10)
    b = np.array([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 3], [2, 1, 3, 0]], dtype=np.uint8)
    boards = np.repeat(b[None], 20, axis=0)
    ve.spawn(boards, rng, kind=True)
    assert set(np.unique(boards[:, 3, 3])) <= {1, 2}
    assert not ve.afterstates(boards)[2].any(axis=1).any()


def test_kind_spawn_keeps_the_ratio_and_touches_only_empty_cells():
    rng = np.random.default_rng(11)
    boards = np.zeros((4000, 4, 4), dtype=np.uint8)
    boards[:, 0, 0] = 3
    before = boards.copy()
    ve.spawn(boards, rng, kind=True)
    diff = (before == 0) & (boards != 0)
    assert (diff.sum(axis=(1, 2)) == 1).all()
    fours = (boards[diff] == 2).mean()
    assert 0.08 < fours < 0.12


def test_vec_env_kind_mode_runs_and_reports_no_dead_boards_prematurely():
    env = ve.VecEnv(8, seed=2, spawn_mode="kind")
    assert env.spawn_mode == "kind"
    for _ in range(200):
        after, rewards, valid = ve.afterstates(env.boards)
        alive = valid.any(axis=1)
        actions = np.where(valid, rewards, -1).argmax(axis=1)
        env.apply(actions, after, rewards, valid)
        env.reset(~alive)
    with pytest.raises(NotImplementedError):
        ve.VecEnv(2, spawn_mode="best")
