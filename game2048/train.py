"""Afterstate TD(0) training of the value network.

    uv run python -m game2048.train --minutes 30

Loop: many boards in parallel; pick moves greedily with the online net; store
(afterstate, next reward, next afterstate, done) in a replay buffer; regress
V(afterstate) toward reward + V_target(next afterstate) (0 when the game
ended), with a random board symmetry applied to each training input. This is
the afterstate TD(0) update of Szubert & Jaskowski (2014) with a CNN instead
of an n-tuple table.
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import nn as g2nn
from . import vec_env as ve


@dataclass
class Config:
    envs: int = 512
    batch: int = 1024
    grad_steps: int = 2
    lr: float = 3e-4
    buffer: int = 1_000_000
    min_buffer: int = 20_000
    tau: float = 0.005
    reward_scale: float = 1 / 2048
    grad_clip: float = 10.0
    iterations: int | None = None
    minutes: float | None = None
    eval_every: int = 2500
    eval_games: int = 200
    log_every: int = 100
    width: int = 128
    hidden: int = 512
    device: str = "auto"
    out: str = "checkpoints"
    seed: int = 0
    resume: str | None = None
    spawn_mode: str = "random"   # "random" or "kind" (see vec_env.spawn)


class ReplayBuffer:
    """(afterstate, scaled reward of the next move, next afterstate, done)."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.after = np.zeros((capacity, 4, 4), dtype=np.uint8)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next = np.zeros((capacity, 4, 4), dtype=np.uint8)
        self.done = np.zeros(capacity, dtype=bool)
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, after: np.ndarray, reward: np.ndarray, nxt: np.ndarray, done: np.ndarray) -> None:
        n = len(after)
        if n == 0:
            return
        if n > self.capacity:
            after, reward, nxt, done = (x[-self.capacity:] for x in (after, reward, nxt, done))
            n = self.capacity
        idx = (self.pos + np.arange(n)) % self.capacity
        self.after[idx] = after
        self.reward[idx] = reward
        self.next[idx] = nxt
        self.done[idx] = done
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, rng: np.random.Generator, batch: int):
        idx = rng.integers(0, self.size, batch)
        return self.after[idx], self.reward[idx], self.next[idx], self.done[idx]


def td_targets(target_model, rewards: np.ndarray, next_after: np.ndarray,
               done: np.ndarray, device) -> torch.Tensor:
    """reward + V_target(next afterstate), or 0 where the game ended."""
    with torch.no_grad():
        v = target_model(g2nn.to_tensor(next_after, device))
    r = torch.from_numpy(rewards.astype(np.float32)).to(device)
    return (r + v).masked_fill(torch.from_numpy(done).to(device), 0.0)


def random_symmetries(boards: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    ks = rng.integers(0, 8, len(boards))
    out = np.empty_like(boards)
    for k in range(8):
        m = ks == k
        if m.any():
            out[m] = ve.apply_symmetry(boards[m], k)
    return out


def evaluate_policy(model, n_games: int, reward_scale: float, device, seed: int,
                    max_steps: int = 100_000, symmetric: bool = False, spawn_mode: str = "random") -> dict:
    env = ve.VecEnv(n_games, seed=seed, spawn_mode=spawn_mode)
    lengths = np.zeros(n_games, dtype=np.int64)
    for _ in range(max_steps):
        actions, after, rewards, valid = g2nn.greedy_actions(
            model, env.boards, device, reward_scale, symmetric=symmetric
        )
        alive = valid.any(axis=1)
        if not alive.any():
            break
        env.apply(actions, after, rewards, valid)
        lengths += alive
    scores, tiles = env.scores, env.max_tiles()
    uniq, counts = np.unique(tiles, return_counts=True)
    return {
        "games": n_games,
        "tile_counts": {int(t): int(c) for t, c in zip(uniq, counts)},
        "mean_score": float(scores.mean()),
        "median_score": float(np.median(scores)),
        "max_score": int(scores.max()),
        "mean_length": float(lengths.mean()),
        "max_tile": int(tiles.max()),
        "rate_1024": float((tiles >= 1024).mean()),
        "rate_2048": float((tiles >= 2048).mean()),
        "rate_4096": float((tiles >= 4096).mean()),
        "rate_8192": float((tiles >= 8192).mean()),
    }


def polyak_update(target, online, tau: float) -> None:
    with torch.no_grad():
        for pt, p in zip(target.parameters(), online.parameters()):
            pt.lerp_(p, tau)


LOG_FIELDS = [
    "iteration", "env_steps", "games", "elapsed_s", "loss", "train_mean_score",
    "train_max_tile", "eval_mean_score", "eval_median_score", "eval_max_score",
    "eval_mean_length", "eval_max_tile", "rate_1024", "rate_2048", "rate_4096", "rate_8192",
]


def _checkpoint_meta(path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True).get("meta", {})


def train(cfg: Config) -> dict:
    """Train; with `cfg.resume` continue that checkpoint's iteration, step and game
    counters and append to an existing log.csv in `cfg.out`."""
    device = g2nn.pick_device(cfg.device)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    resumed = _checkpoint_meta(cfg.resume) if cfg.resume else {}
    it = int(resumed.get("iteration", 0))
    env_steps = int(resumed.get("env_steps", 0))
    games = int(resumed.get("games", 0))
    elapsed_before = float(resumed.get("elapsed_s", 0.0))
    seed = cfg.seed + it
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    if cfg.resume:
        model, _, _ = g2nn.load_checkpoint(cfg.resume, device)
    else:
        model = g2nn.ValueNet(cfg.width, cfg.hidden).to(device)
    target = copy.deepcopy(model)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    buf = ReplayBuffer(cfg.buffer)
    env = ve.VecEnv(cfg.envs, seed=seed, spawn_mode=cfg.spawn_mode)
    idx = np.arange(cfg.envs)

    prev_after = np.zeros_like(env.boards)
    have_prev = np.zeros(cfg.envs, dtype=bool)
    recent_scores: deque = deque(maxlen=500)
    recent_tiles: deque = deque(maxlen=500)
    losses: deque = deque(maxlen=200)
    best_score = -1.0
    if (out / "best.pt").exists():
        best_score = float(_checkpoint_meta(out / "best.pt").get("mean_score", -1.0))
    t0 = time.time()
    elapsed = lambda: time.time() - t0 + elapsed_before  # noqa: E731

    append = bool(cfg.resume) and (out / "log.csv").exists()
    log_file = open(out / "log.csv", "a" if append else "w", newline="")
    log = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
    if not append:
        log.writeheader()

    def mean_loss() -> float:
        return float(torch.stack(list(losses)).mean()) if losses else float("nan")

    def run_eval() -> dict:
        nonlocal best_score
        stats = evaluate_policy(model, cfg.eval_games, cfg.reward_scale, device, seed=cfg.seed + 12345,
                                spawn_mode=cfg.spawn_mode)
        meta = {**stats, "eval_games": stats["games"], "iteration": it, "env_steps": env_steps,
                "games": games, "elapsed_s": elapsed(), "config": dataclasses.asdict(cfg)}
        g2nn.save_checkpoint(out / "latest.pt", model, cfg.reward_scale, meta)
        if stats["mean_score"] > best_score:
            best_score = stats["mean_score"]
            g2nn.save_checkpoint(out / "best.pt", model, cfg.reward_scale, meta)
        log.writerow({
            "iteration": it, "env_steps": env_steps, "games": games,
            "elapsed_s": round(elapsed(), 1), "loss": mean_loss(),
            "train_mean_score": float(np.mean(recent_scores)) if recent_scores else "",
            "train_max_tile": max(recent_tiles) if recent_tiles else "",
            "eval_mean_score": stats["mean_score"], "eval_median_score": stats["median_score"],
            "eval_max_score": stats["max_score"], "eval_mean_length": stats["mean_length"],
            "eval_max_tile": stats["max_tile"], "rate_1024": stats["rate_1024"],
            "rate_2048": stats["rate_2048"], "rate_4096": stats["rate_4096"],
            "rate_8192": stats["rate_8192"],
        })
        log_file.flush()
        print(
            f"[eval] it {it} | {stats['mean_score']:.0f} mean, {stats['median_score']:.0f} median, "
            f"{stats['max_score']} max | tile {stats['max_tile']} | "
            f"1024 {stats['rate_1024']:.0%}, 2048 {stats['rate_2048']:.0%}, "
            f"4096 {stats['rate_4096']:.0%} | best {best_score:.0f}",
            flush=True,
        )
        return stats

    try:
        while True:
            if cfg.iterations is not None and it >= cfg.iterations:
                break
            if cfg.minutes is not None and time.time() - t0 >= cfg.minutes * 60:
                break

            after, rewards, valid = ve.afterstates(env.boards)
            finished = ~valid.any(axis=1)
            ended = have_prev & finished
            if ended.any():  # the previous afterstate led to a dead board
                k = int(ended.sum())
                buf.add(prev_after[ended], np.zeros(k, np.float32),
                        np.zeros((k, 4, 4), np.uint8), np.ones(k, bool))
            if finished.any():
                recent_scores.extend(env.scores[finished].tolist())
                recent_tiles.extend(env.max_tiles()[finished].tolist())
                games += int(finished.sum())
                env.reset(finished)
                after, rewards, valid = ve.afterstates(env.boards)

            v = g2nn.evaluate(model, after.reshape(-1, 4, 4), device).reshape(cfg.envs, 4)
            q = rewards * cfg.reward_scale + v
            q[~valid] = -np.inf
            actions = q.argmax(axis=1)
            chosen = after[idx, actions]
            cont = have_prev & ~finished
            if cont.any():  # previous afterstate -> (spawn) -> this move's reward and afterstate
                buf.add(prev_after[cont], (rewards[idx, actions][cont] * cfg.reward_scale).astype(np.float32),
                        chosen[cont], np.zeros(int(cont.sum()), bool))
            prev_after = chosen
            have_prev[:] = True
            env.apply(actions, after, rewards, valid)
            env_steps += cfg.envs

            if len(buf) >= cfg.min_buffer:
                for _ in range(cfg.grad_steps):
                    a_b, r_b, n_b, d_b = buf.sample(rng, cfg.batch)
                    tgt = td_targets(target, r_b, n_b, d_b, device)
                    pred = model(g2nn.to_tensor(random_symmetries(a_b, rng), device))
                    loss = F.smooth_l1_loss(pred, tgt)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    opt.step()
                    polyak_update(target, model, cfg.tau)
                    losses.append(loss.detach())

            it += 1
            if it % cfg.log_every == 0:
                el = elapsed()
                print(
                    f"it {it} | {env_steps/1e6:.2f}M steps | {games} games | {el:.0f}s "
                    f"| {env_steps/el:.0f} steps/s | train mean {np.mean(recent_scores) if recent_scores else 0:.0f} "
                    f"| max tile {max(recent_tiles) if recent_tiles else 0} | loss {mean_loss():.4f} "
                    f"| buffer {len(buf)}",
                    flush=True,
                )
            if it % cfg.eval_every == 0:
                run_eval()
    except KeyboardInterrupt:
        print("interrupted; running final eval", flush=True)

    if it % cfg.eval_every != 0 or it == 0:
        run_eval()
    log_file.close()
    return {"best_score": best_score, "iterations": it, "env_steps": env_steps,
            "games": games, "best_path": str(out / "best.pt"), "elapsed_s": elapsed()}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in dataclasses.fields(Config):
        kind = {int: int, float: float, str: str}.get(f.type if isinstance(f.type, type) else None)
        if kind is None:  # "int | None" style annotations
            kind = float if "float" in str(f.type) else (str if "str" in str(f.type) else int)
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=kind, default=f.default)
    args = parser.parse_args(argv)
    cfg = Config(**vars(args))
    if cfg.iterations is None and cfg.minutes is None:
        cfg.minutes = 30
    print(cfg, flush=True)
    result = train(cfg)
    print(result, flush=True)


if __name__ == "__main__":
    main()
