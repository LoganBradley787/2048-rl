"""FastAPI server: one game per process, plus a static web UI.

    uvicorn game2048.server:app --reload
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import evaluators
from .agents import AGENTS, Agent, make_agent
from .core import SPAWN_MODES, Direction, Game

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@dataclass
class State:
    game: Game = field(default_factory=Game)
    lock: threading.Lock = field(default_factory=threading.Lock)
    agents: dict[str, Agent] = field(default_factory=dict)
    evaluator: Callable | None = None  # for best/evil spawns; loaded once


def get_agent(name: str) -> Agent:
    """Agents are built once per process; loading a network is not free."""
    if name not in state.agents:
        try:
            state.agents[name] = make_agent(name)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except (FileNotFoundError, ImportError) as e:
            raise HTTPException(status_code=503, detail=str(e))
    return state.agents[name]


state = State()
app = FastAPI(title="2048")


class MoveRequest(BaseModel):
    direction: Direction

    @field_validator("direction", mode="before")
    @classmethod
    def _parse(cls, v):
        if isinstance(v, str) and not v.isdigit():
            try:
                return Direction[v.upper()]
            except KeyError:
                raise ValueError(f"direction must be one of {[d.name.lower() for d in Direction]}")
        return v


class NewGameRequest(BaseModel):
    seed: int | None = None
    mode: Literal["random", "kind", "best", "evil", "choose"] = "random"


class PlaceRequest(BaseModel):
    row: int
    col: int
    value: Literal[2, 4]


class AgentStepRequest(BaseModel):
    agent: str


class AgentRunRequest(BaseModel):
    """Play as many agent steps as fit in `ms` milliseconds (at most `max_steps`)."""
    agent: str
    ms: int = Field(100, ge=1, le=5000)
    max_steps: int = Field(5000, ge=1, le=200_000)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/state")
def get_state():
    with state.lock:
        return state.game.to_dict()


@app.get("/api/modes")
def list_modes():
    return list(SPAWN_MODES)


@app.post("/api/new")
def new_game(req: NewGameRequest | None = None):
    seed = req.seed if req else None
    mode = req.mode if req else "random"
    evaluator = None
    if mode in ("best", "evil", "choose"):
        if state.evaluator is None:
            state.evaluator = evaluators.best_available()
        evaluator = state.evaluator
    with state.lock:
        state.game = Game(seed=seed, spawn_mode=mode, evaluator=evaluator)
        return state.game.to_dict()


@app.post("/api/move")
def move(req: MoveRequest):
    with state.lock:
        if state.game.awaiting_tile:
            raise HTTPException(status_code=409, detail="place a tile before moving again")
        moved, reward = state.game.move(req.direction)
        return {**state.game.to_dict(), "moved": moved, "reward": reward,
                "direction": int(req.direction)}


@app.post("/api/place")
def place(req: PlaceRequest):
    """Choose mode: put the next tile on an empty cell."""
    with state.lock:
        try:
            state.game.place(req.row, req.col, req.value)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
        return state.game.to_dict()


@app.get("/api/agents")
def list_agents():
    return sorted(AGENTS)


def _agent_one_step(agent: Agent, game: Game) -> dict:
    """One agent decision on the game: a move (plus the tile in cool mode), or just
    the tile when a human moved and left it to the agent."""
    if game.awaiting_tile:
        return {"moved": False, "reward": 0, "direction": None, "placed": _agent_place(agent, game)}
    placed = None
    chooser = getattr(agent, "choose", None)
    if chooser is not None and game.spawn_mode == "choose":
        direction, placement = chooser(game)
        moved, reward = game.move(direction)
        if moved:
            game.place(*placement)
            placed = {"row": placement[0], "col": placement[1], "value": placement[2]}
    else:
        direction = agent.act(game)
        moved, reward = game.move(direction)
        if game.awaiting_tile:
            placed = _agent_place(agent, game)
    return {"moved": moved, "reward": reward, "direction": int(direction), "placed": placed}


@app.post("/api/agent/step")
def agent_step(req: AgentStepRequest):
    agent = get_agent(req.agent)
    with state.lock:
        game = state.game
        if game.is_over():
            raise HTTPException(status_code=409, detail="game is over")
        result = _agent_one_step(agent, game)          # step first, then read the state
        return {**game.to_dict(), **result}


@app.post("/api/agent/run")
def agent_run(req: AgentRunRequest):
    """Max-speed play: many steps per request, so the browser is not the bottleneck.
    Returns the state after the batch plus `steps` and the batch's total `reward`."""
    agent = get_agent(req.agent)
    with state.lock:
        game = state.game
        if game.is_over():
            raise HTTPException(status_code=409, detail="game is over")
        deadline = time.monotonic() + req.ms / 1000
        steps, total, last = 0, 0, {"moved": False, "reward": 0, "direction": None, "placed": None}
        while steps < req.max_steps and not game.is_over():
            last = _agent_one_step(agent, game)
            steps += 1
            total += last["reward"]
            if not last["moved"] and last["placed"] is None:
                break                                    # the agent could not act; do not spin
            if time.monotonic() >= deadline:
                break
        return {**game.to_dict(), **last, "steps": steps, "reward": total}


def _agent_place(agent: Agent, game: Game) -> dict:
    """In choose mode an agent without a placement policy gets the best-reply placement."""
    placer = getattr(agent, "place", None)
    row, col, value = placer(game) if placer is not None else game.best_placement()
    game.place(row, col, value)
    return {"row": row, "col": col, "value": value}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
