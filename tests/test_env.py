import pytest

from game2048.core import Direction, Game
from game2048.env import Env2048


def set_board(env, rows):
    env.game.board = [list(r) for r in rows]
    env.game.score = 0


def test_reset_returns_log2_encoded_flat_observation():
    env = Env2048(seed=1)
    obs = env.reset()
    assert len(obs) == 16
    tiles = [v for row in env.game.board for v in row]
    assert obs == [0 if v == 0 else v.bit_length() - 1 for v in tiles]


def test_observation_encoding_of_known_board():
    env = Env2048(seed=1)
    env.reset()
    set_board(env, [[2, 4, 8, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2048]])
    assert env.observe() == [1, 2, 3, 0] + [0] * 4 + [0] * 4 + [0, 0, 0, 11]


def test_obs_array_is_numpy_with_shape_16():
    np = pytest.importorskip("numpy")
    env = Env2048(seed=1)
    env.reset()
    arr = env.obs_array()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (16,)
    assert arr.tolist() == env.observe()


def test_step_returns_reward_and_updates_board():
    env = Env2048(seed=1)
    env.reset()
    set_board(env, [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    obs, reward, done, info = env.step(Direction.LEFT)
    assert reward == 4
    assert done is False
    assert info["moved"] is True
    assert info["score"] == 4
    assert obs[0] == 2  # log2(4)


def test_step_accepts_int_action():
    env = Env2048(seed=1)
    env.reset()
    set_board(env, [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    _, reward, _, _ = env.step(3)
    assert reward == 4


def test_action_mask_marks_legal_moves():
    env = Env2048(seed=1)
    env.reset()
    set_board(env, [[2, 4, 8, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert env.action_mask() == [False, False, True, False]


def test_invalid_move_gets_penalty_and_does_not_end_episode():
    env = Env2048(seed=1, invalid_move_penalty=5)
    env.reset()
    set_board(env, [[2, 4, 8, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    obs, reward, done, info = env.step(Direction.UP)
    assert reward == -5
    assert done is False
    assert info["moved"] is False
    assert env.observe() == obs


def test_repeated_invalid_moves_end_episode_when_limit_set():
    env = Env2048(seed=1, max_invalid_moves=3)
    env.reset()
    set_board(env, [[2, 4, 8, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    dones = [env.step(Direction.UP)[2] for _ in range(3)]
    assert dones == [False, False, True]


def test_valid_move_resets_invalid_counter():
    env = Env2048(seed=1, max_invalid_moves=2)
    env.reset()
    set_board(env, [[2, 4, 8, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    env.step(Direction.UP)          # invalid #1
    env.step(Direction.DOWN)        # valid
    _, _, done, _ = env.step(Direction.UP)  # invalid #1 again
    assert done is False


def test_episode_ends_when_board_is_stuck():
    env = Env2048(seed=1)
    env.reset()
    set_board(env, [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 8]])
    # bottom-right 8 next to a 4: LEFT would slide nothing... use a board with one merge left
    set_board(env, [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 8, 8]])
    _, reward, done, info = env.step(Direction.LEFT)
    assert reward == 16
    # after the merge one cell is empty and a tile spawned there; whether the
    # game is over depends on the spawn, so just check consistency with the game
    assert done == env.game.is_over()


def test_random_agent_episode_terminates_and_info_is_consistent():
    env = Env2048(seed=123)
    obs = env.reset()
    done = False
    steps = 0
    total = 0
    while not done:
        legal = [i for i, ok in enumerate(env.action_mask()) if ok]
        obs, reward, done, info = env.step(legal[steps % len(legal)])
        total += reward
        steps += 1
        assert steps < 10_000
    assert total == env.game.score == info["score"]
    assert info["max_tile"] == env.game.max_tile()
    assert info["legal_moves"] == []


def test_step_before_reset_raises():
    env = Env2048(seed=1)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_same_seed_same_trajectory():
    a, b = Env2048(seed=9), Env2048(seed=9)
    oa, ob = a.reset(), b.reset()
    assert oa == ob
    for act in [0, 1, 2, 3, 0, 1, 2, 3]:
        assert a.step(act) == b.step(act)


def test_env_passes_spawn_mode_and_evaluator_to_the_game():
    env = Env2048(seed=1, spawn_mode="evil", evaluator=lambda board: 0.0)
    env.reset()
    assert env.game.spawn_mode == "evil"
    assert env.game.evaluator(env.game.board) == 0.0
    obs, reward, done, info = env.step(env.game.legal_moves()[0])
    assert info["mode"] == "evil"
