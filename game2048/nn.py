"""Value network over afterstates, and the agent that plays with it.

The network scores a board *after* a move and *before* the random spawn.
Playing is then: for each legal move, reward + V(afterstate); take the max.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import vec_env as ve
from .core import Direction, Game

N_EXPS = 16
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "best.pt"


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ValueNet(nn.Module):
    """One-hot (16 channels) -> two 2x2 convs -> MLP -> scalar value."""

    def __init__(self, width: int = 256, hidden: int = 512) -> None:
        super().__init__()
        self.width, self.hidden = width, hidden
        self.conv1 = nn.Conv2d(N_EXPS, width, kernel_size=2)
        self.conv2 = nn.Conv2d(width, width, kernel_size=2)
        self.fc1 = nn.Linear(width * 2 * 2, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, boards: torch.Tensor) -> torch.Tensor:
        x = F.one_hot(boards, N_EXPS).permute(0, 3, 1, 2).float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x).squeeze(1)


def to_tensor(boards: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(boards, dtype=np.int64)).to(device)


@torch.no_grad()
def evaluate(model: ValueNet, boards: np.ndarray, device, symmetric: bool = False,
             chunk: int = 16384) -> np.ndarray:
    """V for every board, as float32 numpy. `symmetric` averages the 8 symmetries."""
    if symmetric:
        stacked = np.concatenate([ve.apply_symmetry(boards, k) for k in range(8)])
        return evaluate(model, stacked, device).reshape(8, len(boards)).mean(axis=0)
    if len(boards) == 0:
        return np.zeros(0, dtype=np.float32)
    out = [
        model(to_tensor(boards[i:i + chunk], device)).float().cpu().numpy()
        for i in range(0, len(boards), chunk)
    ]
    return np.concatenate(out)


def greedy_actions(model: ValueNet, boards: np.ndarray, device, reward_scale: float,
                   symmetric: bool = False):
    """argmax over legal moves of reward*scale + V(afterstate).

    Returns (actions, after, rewards, valid). Boards with no legal move get
    action 0; check `valid.any(1)` before trusting it.
    """
    after, rewards, valid = ve.afterstates(boards)
    n = len(boards)
    v = evaluate(model, after.reshape(-1, 4, 4), device, symmetric).reshape(n, 4)
    q = rewards * reward_scale + v
    q[~valid] = -np.inf
    return q.argmax(axis=1), after, rewards, valid


def board_to_exps(board: list[list[int]]) -> np.ndarray:
    return np.array([[v.bit_length() - 1 if v else 0 for v in row] for row in board],
                    dtype=np.uint8)[None]


def save_checkpoint(path, model: ValueNet, reward_scale: float, meta: dict) -> None:
    torch.save(
        {"model": model.state_dict(), "width": model.width, "hidden": model.hidden,
         "reward_scale": reward_scale, "meta": meta},
        path,
    )


def load_checkpoint(path, device) -> tuple[ValueNet, float, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = ValueNet(ck["width"], ck["hidden"])
    model.load_state_dict(ck["model"])
    return model.to(device), ck["reward_scale"], ck["meta"]


class NNAgent:
    """Plays with a trained value network. Registered as "nn" in agents.py."""

    def __init__(self, checkpoint=None, device: str = "auto", symmetric: bool = True) -> None:
        path = Path(checkpoint or os.environ.get("GAME2048_CHECKPOINT") or DEFAULT_CHECKPOINT)
        if not path.exists():
            raise FileNotFoundError(
                f"no checkpoint at {path}; train one with `uv run python -m game2048.train`"
            )
        self.device = pick_device(device)
        self.model, self.reward_scale, self.meta = load_checkpoint(path, self.device)
        self.symmetric = symmetric

    def act(self, game: Game) -> Direction:
        actions, _, _, valid = greedy_actions(
            self.model, board_to_exps(game.board), self.device, self.reward_scale, self.symmetric
        )
        if not valid[0].any():
            raise ValueError("no legal moves")
        return Direction(int(actions[0]))
