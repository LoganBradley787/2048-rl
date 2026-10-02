# Using the code

The rules live in one dependency-free Python module, and everything else is
built on it: the web server, the training environment and the agents. Actions
are the same everywhere: `0 = up, 1 = right, 2 = down, 3 = left`.

## Tile modes

The dropdown next to **New game** (and `mode` in `POST /api/new`) picks how
each new tile is placed:

| Mode | The next tile is... |
| --- | --- |
| `random` | the standard game: a uniform empty cell, 2 with probability 0.9, else 4 |
| `kind` | random, but never a placement that ends the game while another placement keeps a move |
| `best` | the placement that maximises your best reply (reward plus the evaluator's value of the resulting board) |
| `evil` | the placement that minimises it, and ends the game whenever a tile can |
| `choose` | cool mode: you place it yourself, any empty cell, 2 or 4 |

`best` and `evil` judge boards with the trained n-tuple tables when
`checkpoints/ntuple/weights.bin` exists, else with a small heuristic based on
empty cells and merges.

## The game in Python

```python
from game2048.core import Direction, Game

game = Game(seed=0)                  # spawn_mode="random" by default
moved, reward = game.move(Direction.LEFT)
game.board                           # 4 lists of 4 tile values, 0 for empty
game.legal_moves()                   # the directions that change the board
game.is_over(), game.score, game.max_tile()
copy = game.clone()                  # for search
```

In cool mode (`Game(spawn_mode="choose")`) a move leaves the game waiting for
a tile, and `game.place(row, col, 2 or 4)` puts it down.

## Training against the environment

`Env2048` wraps `Game` in the usual `reset()` / `step(action)` loop:

```python
from game2048.env import Env2048

env = Env2048(seed=0, invalid_move_penalty=1.0, max_invalid_moves=10)
obs = env.reset()                 # list of 16 ints: log2(tile), 0 for empty
done = False
while not done:
    mask = env.action_mask()      # [bool] * 4, True where the move changes the board
    action = policy(obs, mask)    # your network goes here
    obs, reward, done, info = env.step(action)
    # reward = points from merges this move, or -invalid_move_penalty
    # info = {"moved", "score", "max_tile", "legal_moves", "mode"}
```

`env.obs_array()` returns the observation as a numpy int64 array of shape
`(16,)`. `env.game` is the underlying `Game`. With a seed, every `reset()`
starts a different game and the whole sequence of games is reproducible.
`Env2048(spawn_mode=...)` takes the tile modes above; in `choose` mode call
`env.game.place(...)` after each step.

For training at scale, `game2048/vec_env.py` runs many boards at once in
numpy (`VecEnv`, spawn modes `random` and `kind`). That is what the CNN
trainer uses.

## Watching your own agent in the browser

Anything with an `act(game) -> Direction` method is an agent. Register it in
`game2048/agents.py`:

```python
class MyNetAgent:
    def __init__(self):
        self.model = load_my_model()

    def act(self, game):
        obs = [v.bit_length() - 1 if v else 0 for row in game.board for v in row]
        legal = game.legal_moves()
        scores = self.model(obs)
        return max(legal, key=lambda d: scores[int(d)])

AGENTS["mynet"] = MyNetAgent
```

Restart the server and "mynet" shows up in the dropdown. An agent that also
has `choose(game) -> (Direction, (row, col, 2 | 4))` plays cool mode by
picking the move and the tile together. Otherwise the server places the best
tile for it (`Game.best_placement`).

## HTTP API

The server holds one game per process. Every response is the game state,
sometimes with extra fields.

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/api/state` | | `{board, score, over, legal_moves, max_tile, mode, awaiting_tile}` |
| POST | `/api/new` | `{"seed": int, "mode": "random"}` (both optional) | fresh state |
| POST | `/api/move` | `{"direction": 0..3 or "up"/"right"/"down"/"left"}` | state plus `{moved, reward, direction}` |
| POST | `/api/place` | `{"row", "col", "value": 2 or 4}` (cool mode, after a move) | state |
| GET | `/api/modes` | | list of tile modes |
| GET | `/api/agents` | | list of agent names |
| POST | `/api/agent/step` | `{"agent": "greedy"}` | state plus `{moved, reward, direction}`, and `placed` in cool mode |
| POST | `/api/agent/run` | `{"agent": "snake", "ms": 100, "max_steps": 5000}` | as many steps as fit in `ms`: state plus `steps` and the batch's total `reward` |

The agent endpoints answer 404 for an unknown agent, 409 when the game is
over, and 503 when the agent cannot load (no trained weights yet, or PyTorch
is not installed for `nn`). The **Max speed** checkbox in the UI uses
`/api/agent/run`, so the browser redraws once per batch instead of once per
move.

## Environment variables

| Variable | Overrides |
| --- | --- |
| `GAME2048_NTUPLE` | weights for the `ntuple` agent (default `checkpoints/ntuple/weights.bin`) |
| `GAME2048_NTUPLE_COOL` | weights for `ntuple-cool` (default `checkpoints/ntuple_choose/weights.bin`) |
| `GAME2048_CHECKPOINT` | checkpoint for `nn` (default `checkpoints/best.pt`) |
| `CC` | the C compiler used to build the engine (default `cc`) |
