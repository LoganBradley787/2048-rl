# 2048-rl

2048-rl is a set of 2048 players. The strong one against random spawns is an
n-tuple network written in C that learns by afterstate temporal-difference
self-play, tens of billions of moves in a few hours on a laptop, and plays with
expectimax. In "cool mode", where the player places its own tiles, the game is
a deterministic puzzle, and a beam search guided by a snake-order heuristic
plays it to the 131072 tile and 3.67M points, about 93% of the theoretical
maximum, with no training at all. A CNN value network is kept as a baseline.
The game rules, a FastAPI server and a plain HTML/JS board come with it, so you
can play by hand, try cool mode yourself, or watch the agents at max speed.

The rules and state are entirely in Python. The same code doubles as a
gym-style environment, and any agent with an `act(game)` method shows up in
the browser.

## Run the web UI

```bash
uv sync --extra dev
```

```bash
uv run uvicorn game2048.server:app --reload --port 8048
```

Then open http://localhost:8048. Arrow keys or WASD move; swipe on touch.
The **Step** and **Auto-play** buttons drive the game with the agent chosen
in the dropdown. **Max speed** makes auto-play batch about 100 ms of moves per
request (`POST /api/agent/run`) and redraw once per batch, so a full cool-mode
game takes minutes rather than most of an hour; the speed readout then shows
the moves per second achieved.

If you would rather not use uv, `pip install -e ".[dev]"` and
`python -m uvicorn game2048.server:app --port 8048` do the same thing.

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
empty cells and merges. The same modes work for training: `Env2048(spawn_mode=...)`
and `Config(spawn_mode=...)` for the CNN trainer (`random` and `kind` in the
vectorised engine; all of them in the Python `Game`).

## Layout

| Path | What it is |
| --- | --- |
| `game2048/core.py` | Rules engine. `Game` holds the board and score, `move()` slides/merges/spawns. No dependencies. |
| `game2048/env.py` | `Env2048`: `reset()` / `step(action)` wrapper for training. |
| `game2048/agents.py` | `Agent` protocol, `RandomAgent`, `GreedyAgent`, and the `AGENTS` registry the server reads. |
| `game2048/server.py` | FastAPI app and JSON API. One game per process. |
| `game2048/vec_env.py` | numpy engine for many boards at once, used by training and evaluation. |
| `game2048/nn.py` | `ValueNet`, checkpoint helpers, and `NNAgent` (the "nn" entry in the UI). |
| `game2048/train.py` | Afterstate TD(0) trainer; `python -m game2048.train`. |
| `examples/evaluate.py` | Scores a checkpoint over many games and prints the max-tile histogram. |
| `game2048/native/ntuple.c` | Bitboard engine, n-tuple network, TD/TC learning, expectimax and beam search, all multithreaded. |
| `game2048/ntuple.py` | ctypes wrapper: `NTupleNet`, `NTupleAgent` (the "ntuple" and "ntuple-cool" entries in the UI). |
| `game2048/ntuple_train.py` | Training CLI for the n-tuple network; `python -m game2048.ntuple_train`. |
| `examples/evaluate_ntuple.py` | Evaluates n-tuple weights at several depths, in random mode or cool mode. |
| `static/` | The browser UI. |
| `examples/rollout.py` | Runs episodes with a named agent and prints stats. |
| `tests/` | pytest suite for the rules, env, agents, API, and the C engine. |

Actions are the same everywhere: `0 = up, 1 = right, 2 = down, 3 = left`.

## Training against the environment

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
    # info = {"moved", "score", "max_tile", "legal_moves"}
```

`env.obs_array()` returns the observation as a numpy int64 array of shape
`(16,)`. `env.game` is the underlying `Game` if you need the raw board or
want to `clone()` it for search.

The `--extra train` group installs numpy. Add torch or whatever you train
with alongside it.

## Watching a trained agent in the browser

Register anything with an `act(game) -> Direction` method in
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
has `choose(game) -> (Direction, (row, col, 2 | 4))` plays cool mode by picking
the move and the tile together; otherwise the server places the best tile
for it (`Game.best_placement`).

## API

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/api/state` | | `{board, score, over, legal_moves, max_tile, mode, awaiting_tile}` |
| POST | `/api/move` | `{"direction": 0..3 or "up"/"right"/"down"/"left"}` | state plus `{moved, reward, direction}` |
| POST | `/api/place` | `{"row", "col", "value": 2 or 4}` (cool mode, after a move) | state |
| POST | `/api/new` | `{"seed": int, "mode": "random"}` (both optional) | fresh state |
| GET | `/api/modes` | | list of tile modes |
| GET | `/api/agents` | | list of agent names |
| POST | `/api/agent/step` | `{"agent": "greedy"}` | state plus `{moved, reward, direction}`, and `placed` in cool mode; 404 unknown agent, 409 if the game is over |
| POST | `/api/agent/run` | `{"agent": "ntuple-cool", "ms": 100, "max_steps": 5000}` | as many steps as fit in `ms`; state plus `steps` and the batch's total `reward` |

## Training a network

```bash
uv sync --extra dev --extra train
```

```bash
uv run python -m game2048.train --minutes 30
```

This trains a value network with afterstate TD(0) and writes to `checkpoints/`:
`best.pt` (highest evaluation mean score), `latest.pt`, and `log.csv` with one
row per evaluation. Once `best.pt` exists, pick **nn** in the web UI dropdown
to watch it play (no restart needed; the server loads it on first use).

How it works: the network scores a board *after* a move and *before* the
random spawn, V(afterstate). To play, it tries the four moves and takes the
one with the highest reward + V. To learn, it regresses V(afterstate) toward
reward + V(next afterstate), or 0 when the game ended, using a replay buffer,
a slowly tracking target network, and a random board symmetry on each sample.
The vectorised numpy engine in `game2048/vec_env.py` runs hundreds of games
in parallel with a 65536-entry row lookup table.

Useful flags: `--envs`, `--batch`, `--grad-steps`, `--lr`, `--width`,
`--hidden`, `--eval-every`, `--eval-games`, `--device cpu|mps|cuda`,
`--resume checkpoints/latest.pt`. Run with `--help` for all of them.

Evaluate a checkpoint over many games:

```bash
uv run python examples/evaluate.py --games 500 --symmetric
```

`--symmetric` averages V over the eight board symmetries at play time, which
the **nn** agent in the UI also does.

## The strong player: n-tuple network + expectimax

The CNN above tops out around 74% at 2048. The strongest known approach to
2048 is an n-tuple network, a set of lookup tables over fixed cell patterns
(here four 6-cell patterns under all 8 board symmetries, 136M weights),
trained with the same afterstate TD(0) idea and searched with expectimax at
play time. That lives in `game2048/native/ntuple.c` and is driven from Python
through `game2048/ntuple.py`. The C side has 128-bit bitboards with five bits
per cell, so tiles go all the way to 131072, lock-free multithreaded learning
with temporal-coherence learning rates, multi-stage weight tables, expectimax
with a transposition table, and the cool-mode searches described below. It
compiles itself with the system C compiler on first use.

```bash
uv run python -m game2048.ntuple_train --hours 2 --threads 12
```

```bash
uv run python -m game2048.ntuple_train --hours 6 --threads 12 --resume checkpoints/ntuple/latest.bin --stages 16384,24576,32768 --tc-stages 1 --alpha-plain 0.025 --lock-memory --out checkpoints/ntuple_stages
```

Training ran at 6 to 7 million moves per second on an M3 Pro with the earlier
64-bit engine. `--stages` gives boards whose tile mass (sum of all tiles) has
crossed each boundary their own weight tables, the multi-stage trick behind
the best published 32768 rates; `--resume single-stage.bin --stages ...` seeds
stage 0 from an existing run. Temporal-coherence learning state costs about
1.1 GB per stage on top of the weights, so `--tc-stages 1` keeps it on stage 0
only and trains the other stages with plain TD at `--alpha-plain` (544 MB each
plus a touched bit per entry); `--no-tc` drops it everywhere. `--lock-memory`
pins the tables in RAM so paging cannot stall the lookups. `--patterns 8x6`
starts a new network with eight 6-cell patterns instead of four (twice the
capacity, 1.1 GB of weights). Weights and counters go to `--out`; `latest.bin`
resumes training, `weights.bin` is the smaller deployment copy the UI loads.
Weight files written before the 128-bit engine are converted when loaded.

Evaluate at several search depths and build the report page:

```bash
uv run python examples/evaluate_ntuple.py --games 100 --depths 0,1,2,3 --json checkpoints/ntuple/evals.json
```

```bash
uv run python -c "from game2048.report import ntuple_main; ntuple_main()"
```

Pick **ntuple** in the web UI to watch it play with depth-3 expectimax
(about 8 ms per move). `GAME2048_NTUPLE` overrides the weights path. In
Python:

```python
from game2048 import ntuple as nt
net = nt.NTupleNet.load("checkpoints/ntuple/latest.bin")
stats = nt.summarize(net.play(games=100, depth=3, threads=12))
```

`depth` counts random-spawn layers searched; depth 2 is the "3-ply"
expectimax of the literature.

### Cool mode: tables trained for placing the tiles

In cool mode there is no chance node, so the search maximises over moves and
over placements (`NTupleNet.best_choose`, `choose_value`, `play_choose`;
`depth` counts placement decisions and `topk` placements are kept per node).
Tables trained on random spawns never saw a 32768, so they stall there in
cool mode. `--choose` trains tables by self-play in the choose game instead,
TD(0) on afterstates with the depth-1 choose search as the policy, and
`--explore 0.02` places a random tile on 2% of steps so the games do not all
replay one line:

```bash
uv run python -m game2048.ntuple_train --choose --patterns 8x6 --hours 3 --threads 12 --lock-memory --out checkpoints/ntuple_choose --chunk 1000 --explore 0.02
```

Choose-mode games run thousands of moves with a small search each, so use a
small `--chunk`. Leave `--max-moves` at its default. Games cut at a cap leave
their late boards ungrounded, the values inflate, and the policy collapses.
Pick **ntuple-cool** in the web UI (weights from `GAME2048_NTUPLE_COOL`, else
`checkpoints/ntuple_choose/weights.bin`); it answers 503 until the weights exist.

With no chance nodes the choose game is a deterministic puzzle, and beam
search suits it better than a tree: keep the `width` best boards, extend each
by a move and a placement, repeat `depth` times, then play the first step of
the best line (`NTupleNet.beam_plan`, `beam_choose`, `beam_value`,
`play_beam`). An entry is ranked by its rewards so far plus the best next
move's reward and value. With unlimited width this is exactly the max-max tree
of the same depth, and a test checks that. `spread` caps how many survivors
may share a parent, `stride` commits several steps of a plan before searching
again, and `snake` adds a weighted bonus for keeping the chain laid out as a
snake along the board. On the 5-minute cool-mode tables, 6 games each, with
32768 reached in every game:

| search | mean score | cost per move |
| --- | --- | --- |
| max-max tree, depth 3 | 599k | 9 ms |
| beam, width 16, depth 16 | 369k (stalls at 16384) | 16 ms |
| beam, width 32, depth 12 | 808k | 20 ms |

Width matters more than depth: a narrow beam fills up with variants of one
line. The **ntuple-cool** agent uses width 32 and depth 12 and plays four
steps of each plan before planning again. A deterministic policy makes the
choose game a single canonical line. Different start tiles, and even a prefix
of random spawns (`prefix=`), converge to the same game within a few hundred
moves, so two evaluation games are as good as twelve, and small score
differences between search settings are chaotic rather than statistical.

### The snake player: no tables, 131072

The learned tables turned out to be the weak link in the endgame, where a
16-tile chain has to be collapsed in exact order. The **snake** agent drops
them entirely: an untouched network (value 0 everywhere) so the beam ranks
boards by their merges plus the snake heuristic alone, with the decay set to
0.9 so every tile on the path counts (`nt.set_snake_decay`), 128 lines, 16
steps ahead, four steps per plan. On the canonical game:

| evaluator | beam | score | max tile | moves | time |
| --- | --- | --- | --- | --- | --- |
| best learned tables, snake 0 | 32 x 12 | 1,528,428 | 65536 | 35,686 | 3 min |
| snake only, decay 0.5 | 32 x 12 | 785,284 | 32768 | 16,405 | 37 s |
| snake only, decay 0.5 | 64 x 16 | 1,614,376 | 65536 | 30,801 | 2.4 min |
| snake only, decay 0.5 | 128 x 16 | 1,704,108 | 65536 | 32,806 | 2.3 min |
| snake only, decay 0.75 | 128 x 16 | 3,670,436 | **131072** | 65,636 | 4.3 min |
| snake only, decay 0.9 | 128 x 16 | **3,670,068** | **131072** | 65,544 | 4.2 min |

The theoretical maximum on a 4x4 board is about 3.93M, so the snake player is
within 7% of perfect play. Pick **snake** in the web UI with Cool mode and Max
speed to watch it.

```python
net = nt.NTupleNet.load("checkpoints/ntuple_choose/weights.bin")
stats = nt.summarize(net.play_beam(games=2, width=32, depth=12, stride=4, threads=2))
```

## Results

Trained on an M3 Pro (MPS) for 46 minutes: 25 minutes at learning rate 3e-4,
then 21 minutes resumed from the best checkpoint at 1e-4. 53.3M moves,
54,645 games. The checkpoint that ships in `checkpoints/best.pt` is from
iteration 95,000. Evaluated over 500 fresh games (`--seed 999`):

| Player | Mean score | Median | Best game | Reach 1024 | Reach 2048 | Reach 4096 |
| --- | --- | --- | --- | --- | --- | --- |
| Random moves | 1,197 | | | 0% | 0% | 0% |
| Greedy (max immediate merge) | 3,079 | | | 0% | 0% | 0% |
| Value net | 23,392 | 23,236 | | 86.6% | 57.2% | 6.6% |
| Value net, 8-symmetry average (the UI's **nn** agent) | 29,869 | 30,984 | 76,532 | 94.6% | 73.6% | 14.0% |

Two n-tuple networks, each evaluated over the same 100 games per depth
(`--seed 2024`):

| Network | Search depth | Mean score | Median | Best game | Reach 8192 | Reach 16384 | Reach 32768 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 patterns, 2 h, 4.5M games | 0 | 241,803 | 270,258 | 372,332 | 91% | 61% | 0% |
| | 2 | 343,454 | 362,828 | 385,160 | 99% | 95% | 0% |
| | 3 | 345,675 | 363,168 | 385,588 | 100% | 97% | 0% |
| **8 patterns, 4 h, 4.0M games** | 0 | 276,699 | 290,356 | 375,016 | 97% | 75% | 0% |
| | 2 | 342,382 | 362,248 | 528,284 | 100% | 93% | 1% |
| | 3 (the UI's **ntuple** agent) | **351,671** | **365,754** | **621,048** | 99% | 98% | 1% |

The 4-pattern tables top out near 385,000, the score of a board holding
16384, 8192 and the rest, because they never learn to merge two 16384s. The
8-pattern network (`--patterns 8x6`) has the capacity for that and is the
only one that reaches 32768; it ships as `checkpoints/ntuple/weights.bin`
(the 4-pattern weights are kept as `weights_4x6.bin`). A 3-stage run seeded
from the 4-pattern tables (`--stages 16384,24576`) scored 313,338 at depth 2
and was dropped.

In cool mode, 8-pattern tables trained for 4 hours on the choose game and
played with the beam (width 32, depth 12, four steps per plan) scored
1,299,016 on the canonical game and built a 65536.

Regenerate the report page from the log and an evaluation:

```bash
uv run python examples/evaluate.py --games 500 --symmetric --json checkpoints/eval.json
```

```bash
uv run python -m game2048.report --log checkpoints/log.csv --eval checkpoints/eval.json --out checkpoints/report.html
```

## Tests

```bash
uv run pytest
```

## Example rollout

```bash
uv run python examples/rollout.py --agent greedy --episodes 20
```
