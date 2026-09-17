"""FastAPI server: one game per process, plus a static web UI.

    uvicorn game2048.server:app --reload
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

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


@app.post("/api/agent/step")
def agent_step(req: AgentStepRequest):
    agent = get_agent(req.agent)
    with state.lock:
        game = state.game
        if game.is_over():
            raise HTTPException(status_code=409, detail="game is over")
        placed = None
        if game.awaiting_tile:  # a human moved and left the tile to the agent
            placed = _agent_place(agent, game)
            return {**game.to_dict(), "moved": False, "reward": 0, "direction": None, "placed": placed}
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
        return {**game.to_dict(), "moved": moved, "reward": reward,
                "direction": int(direction), "placed": placed}


def _agent_place(agent: Agent, game: Game) -> dict:
    """In choose mode an agent without a placement policy gets the best-reply placement."""
    placer = getattr(agent, "place", None)
    row, col, value = placer(game) if placer is not None else game.best_placement()
    game.place(row, col, value)
    return {"row": row, "col": col, "value": value}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
