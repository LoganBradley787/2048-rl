import csv

import numpy as np
import torch

from game2048 import nn as g2nn
from game2048 import train as tr


def test_replay_buffer_wraps_and_samples_only_filled_slots():
    buf = tr.ReplayBuffer(capacity=5)
    after = np.arange(7 * 16, dtype=np.uint8).reshape(7, 4, 4) % 13
    nxt = after[::-1].copy()
    reward = np.arange(7, dtype=np.float32)
    done = np.array([0, 0, 1, 0, 0, 1, 0], dtype=bool)
    buf.add(after[:3], reward[:3], nxt[:3], done[:3])
    assert len(buf) == 3
    buf.add(after[3:], reward[3:], nxt[3:], done[3:])
    assert len(buf) == 5
    a, r, n, d = buf.sample(np.random.default_rng(0), 64)
    assert a.shape == (64, 4, 4) and n.shape == (64, 4, 4)
    assert r.shape == (64,) and d.shape == (64,)
    # everything sampled must be one of the last five entries, with its own reward
    kept = {after[i].tobytes(): float(reward[i]) for i in range(2, 7)}
    assert all(kept[x.tobytes()] == float(rw) for x, rw in zip(a, r))


def test_td_targets_are_zero_when_done_and_bootstrapped_otherwise():
    model = g2nn.ValueNet(width=8, hidden=16)
    for p in model.parameters():
        p.data.zero_()
    model.fc2.bias.data.fill_(2.0)  # V(anything) == 2
    nxt = np.zeros((2, 4, 4), dtype=np.uint8)
    reward = np.array([3.0, 3.0], dtype=np.float32)
    done = np.array([False, True])
    tgt = tr.td_targets(model, reward, nxt, done, device="cpu")
    assert tgt.tolist() == [5.0, 0.0]


def test_evaluate_policy_returns_stats():
    torch.manual_seed(0)
    model = g2nn.ValueNet(width=8, hidden=16)
    stats = tr.evaluate_policy(model, n_games=4, reward_scale=1 / 2048, device="cpu", seed=0)
    assert stats["mean_score"] > 0
    assert stats["games"] == 4
    assert 0 <= stats["rate_2048"] <= 1
    assert stats["max_tile"] >= 8
    assert sum(stats["tile_counts"].values()) == 4
    assert max(stats["tile_counts"]) == stats["max_tile"]


def test_evaluate_policy_can_average_symmetries():
    torch.manual_seed(0)
    model = g2nn.ValueNet(width=8, hidden=16)
    stats = tr.evaluate_policy(model, n_games=3, reward_scale=1 / 2048, device="cpu", seed=0,
                               symmetric=True)
    assert stats["games"] == 3 and stats["mean_score"] > 0


def test_train_smoke_writes_log_and_checkpoints(tmp_path):
    cfg = tr.Config(
        envs=8, batch=16, grad_steps=1, min_buffer=32, iterations=40, eval_every=20,
        eval_games=4, width=8, hidden=16, device="cpu", out=str(tmp_path), seed=0,
        log_every=10,
    )
    result = tr.train(cfg)
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "latest.pt").exists()
    rows = list(csv.DictReader(open(tmp_path / "log.csv")))
    assert len(rows) >= 2
    assert all(np.isfinite(float(r["loss"])) for r in rows if r["loss"])
    assert result["best_score"] > 0
    loaded, scale, meta = g2nn.load_checkpoint(tmp_path / "best.pt", "cpu")
    assert scale == cfg.reward_scale and "iteration" in meta
    # training counters must not be overwritten by the evaluation's own "games" count
    assert meta["games"] == result["games"] and meta["eval_games"] == cfg.eval_games


def test_resume_continues_counters_and_appends_to_log(tmp_path):
    import dataclasses

    cfg = tr.Config(
        envs=8, batch=16, grad_steps=1, min_buffer=32, iterations=20, eval_every=10,
        eval_games=4, width=8, hidden=16, device="cpu", out=str(tmp_path), seed=0, log_every=10,
    )
    first = tr.train(cfg)
    rows1 = list(csv.DictReader(open(tmp_path / "log.csv")))
    assert [int(float(r["iteration"])) for r in rows1] == [10, 20]

    cfg2 = dataclasses.replace(cfg, resume=str(tmp_path / "latest.pt"), iterations=40)
    second = tr.train(cfg2)
    rows2 = list(csv.DictReader(open(tmp_path / "log.csv")))
    assert [int(float(r["iteration"])) for r in rows2] == [10, 20, 30, 40]
    assert rows2[:2] == rows1  # the earlier rows are kept, not regenerated
    assert open(tmp_path / "log.csv").read().count("iteration,") == 1  # header written once
    assert int(float(rows2[-1]["env_steps"])) == 40 * cfg.envs
    assert float(rows2[-1]["elapsed_s"]) >= float(rows2[1]["elapsed_s"])
    assert second["iterations"] == 40 and second["env_steps"] == 40 * cfg.envs
    # best.pt is only replaced when the resumed run beats the earlier best
    assert second["best_score"] >= first["best_score"]


def test_train_smoke_with_kind_spawns(tmp_path):
    cfg = tr.Config(
        envs=8, batch=16, grad_steps=1, min_buffer=32, iterations=20, eval_every=10,
        eval_games=4, width=8, hidden=16, device="cpu", out=str(tmp_path), seed=0,
        log_every=10, spawn_mode="kind",
    )
    result = tr.train(cfg)
    assert result["best_score"] > 0
    _, _, meta = g2nn.load_checkpoint(tmp_path / "best.pt", "cpu")
    assert meta["config"]["spawn_mode"] == "kind"
