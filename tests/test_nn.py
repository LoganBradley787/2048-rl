import numpy as np
import pytest
import torch

from game2048 import nn as g2nn
from game2048 import vec_env as ve
from game2048.core import Game


def zeroed(width=8, hidden=16):
    model = g2nn.ValueNet(width=width, hidden=hidden)
    for p in model.parameters():
        p.data.zero_()
    return model


def rand_boards(seed, n):
    rng = np.random.default_rng(seed)
    b = rng.integers(1, 10, size=(n, 4, 4)).astype(np.uint8)
    b[rng.random((n, 4, 4)) > 0.6] = 0
    return b


def test_value_net_maps_boards_to_scalars():
    model = g2nn.ValueNet(width=8, hidden=16)
    x = g2nn.to_tensor(rand_boards(0, 5), "cpu")
    assert x.dtype == torch.int64 and x.shape == (5, 4, 4)
    assert model(x).shape == (5,)


def test_evaluate_returns_numpy_of_length_n():
    model = g2nn.ValueNet(width=8, hidden=16)
    v = g2nn.evaluate(model, rand_boards(1, 7), "cpu")
    assert isinstance(v, np.ndarray) and v.shape == (7,)


def test_symmetric_evaluate_is_invariant_under_all_eight_symmetries():
    torch.manual_seed(0)
    model = g2nn.ValueNet(width=8, hidden=16)
    b = rand_boards(2, 6)
    base = g2nn.evaluate(model, b, "cpu", symmetric=True)
    for k in range(8):
        v = g2nn.evaluate(model, ve.apply_symmetry(b, k), "cpu", symmetric=True)
        np.testing.assert_allclose(v, base, rtol=1e-5, atol=1e-5)


def test_greedy_actions_only_returns_valid_moves():
    torch.manual_seed(1)
    model = g2nn.ValueNet(width=8, hidden=16)
    b = rand_boards(3, 50)
    actions, after, rewards, valid = g2nn.greedy_actions(model, b, "cpu", reward_scale=1 / 2048)
    assert actions.shape == (50,)
    alive = valid.any(1)
    assert valid[np.arange(50), actions][alive].all()


def test_greedy_actions_with_zero_value_picks_max_reward():
    model = zeroed()
    b = np.array([[[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [3, 3, 0, 0]]], dtype=np.uint8)
    actions, *_ = g2nn.greedy_actions(model, b, "cpu", reward_scale=1 / 2048)
    assert actions[0] in (1, 3)  # LEFT/RIGHT merge 2+2 and 8+8; UP/DOWN merge nothing


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(2)
    model = g2nn.ValueNet(width=8, hidden=16)
    path = tmp_path / "m.pt"
    g2nn.save_checkpoint(path, model, reward_scale=0.25, meta={"step": 7})
    loaded, scale, meta = g2nn.load_checkpoint(path, "cpu")
    assert scale == 0.25 and meta["step"] == 7
    b = rand_boards(4, 3)
    np.testing.assert_allclose(g2nn.evaluate(loaded, b, "cpu"), g2nn.evaluate(model, b, "cpu"))


def test_nn_agent_plays_a_full_game_with_legal_moves(tmp_path):
    torch.manual_seed(3)
    path = tmp_path / "m.pt"
    g2nn.save_checkpoint(path, g2nn.ValueNet(width=8, hidden=16), reward_scale=1 / 2048, meta={})
    agent = g2nn.NNAgent(checkpoint=path, device="cpu")
    game = Game(seed=5)
    steps = 0
    while not game.is_over():
        d = agent.act(game)
        assert d in game.legal_moves()
        game.move(d)
        steps += 1
        assert steps < 5000
    assert game.score > 0


def test_nn_agent_missing_checkpoint_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train"):
        g2nn.NNAgent(checkpoint=tmp_path / "nope.pt", device="cpu")
