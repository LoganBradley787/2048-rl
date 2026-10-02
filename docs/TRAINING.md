# Training

Trained weights are not in the repository (the n-tuple tables are 0.5 to 3 GB
per run). Everything below writes to `checkpoints/`, which is gitignored, and
the web UI picks the files up from there with no restart.

## The n-tuple network

```bash
uv run python -m game2048.ntuple_train --hours 2 --threads 12
```

The first run compiles `game2048/native/ntuple.c` with your system C compiler.
It writes to `checkpoints/ntuple/`:

| File | What it is |
| --- | --- |
| `latest.bin` | weights plus learning state; pass it to `--resume` to continue |
| `weights.bin` | weights only, a third of the size; this is what the UI loads |
| `state.json` | counters and the settings of the run |
| `log.csv` | one row per chunk of training games |
| `eval.csv` | the periodic expectimax evaluations |

Pick **ntuple** in the web UI to watch it play with depth-3 expectimax, about
8 ms per move.

Ten minutes is enough for a player that reaches 8192 in nearly every game
(the README has the numbers). The two runs behind the results table:

```bash
uv run python -m game2048.ntuple_train --hours 2 --threads 12
```

```bash
uv run python -m game2048.ntuple_train --hours 4 --threads 12 --patterns 8x6
```

### Flags worth knowing

- `--patterns 8x6` starts a network with eight 6-cell patterns instead of four:
  twice the capacity, 1.1 GB of weights, and the only one here that reaches 32768.
- `--no-tc` uses plain TD instead of temporal coherence learning. Temporal
  coherence keeps two extra numbers per weight and gives every weight its own
  learning rate, which falls as that weight's updates start to cancel out.
  It costs three times the memory while training.
- `--stages 16384,24576,32768` gives boards whose tile mass (the sum of all
  tiles) has crossed each boundary their own weight tables. This is the
  multi-stage idea from the literature. `--resume single-stage.bin --stages ...`
  seeds stage 0 from an existing run. It did not help here; see the README.
- `--tc-stages 1 --alpha-plain 0.025` keeps the temporal coherence state on
  stage 0 only and trains the later stages with plain TD, which saves about
  1.1 GB per stage.
- `--lock-memory` pins the tables in RAM so paging cannot stall the lookups.
- `--save-every-min`, `--eval-every-min`, `--eval-games`, `--eval-depth`
  control checkpoints and the evaluations logged to `eval.csv`.

Run with `--help` for the rest. Weight files written by earlier versions of
the engine are converted when loaded.

### Evaluate and build the report page

```bash
uv run python examples/evaluate_ntuple.py --games 100 --depths 0,1,2,3 --json checkpoints/ntuple/evals.json
```

```bash
uv run python -m game2048.report --ntuple
```

That writes `checkpoints/ntuple/report.html`, a self-contained page with the
learning curve and the evaluation at each depth. `depth` counts the layers of
random spawns searched, so depth 2 is the "3-ply" expectimax of the papers.

### From Python

```python
from game2048 import ntuple as nt

net = nt.NTupleNet()                                  # four 6-cell patterns
net.train(games=20_000, threads=12)                   # self-play, returns per-game stats
stats = nt.summarize(net.play(games=100, depth=2, threads=12))
net.save("checkpoints/ntuple/latest.bin")

net = nt.NTupleNet.load("checkpoints/ntuple/latest.bin")
move = net.best_move(nt.tiles_to_bits(game.board), depth=3)
```

## Tables for cool mode

In cool mode the player places every tile, so training by self-play needs the
choose game instead of random spawns:

```bash
uv run python -m game2048.ntuple_train --choose --patterns 8x6 --hours 3 --threads 12 --lock-memory --out checkpoints/ntuple_choose --chunk 1000 --explore 0.02
```

- Use a small `--chunk`: choose-mode games run for thousands of moves each.
- `--explore 0.02` places a random tile on 2% of steps. Without it every
  training game replays the same line, because the game is deterministic.
- Leave `--max-moves` at its default. Games cut at a cap leave their late
  boards without a terminal update, the values inflate, and the policy collapses.

Pick **ntuple-cool** in the web UI to watch the result. These tables turned
out to be weaker than the search-only `snake` agent; [COOL_MODE.md](COOL_MODE.md)
has the story.

## The CNN baseline

This one needs PyTorch:

```bash
uv sync --extra dev --extra train
```

```bash
uv run python -m game2048.train --minutes 30
```

It trains a small convolutional value network with the same afterstate TD(0)
idea and writes `checkpoints/best.pt` (highest evaluation mean score),
`latest.pt`, and `log.csv` with one row per evaluation. Pick **nn** in the web
UI once `best.pt` exists.

The trainer runs hundreds of games in parallel on the numpy engine in
`game2048/vec_env.py`, stores transitions in a replay buffer, regresses
V(afterstate) toward reward + V(next afterstate) from a slowly tracking target
network, and applies a random board symmetry to each sample.

Useful flags: `--envs`, `--batch`, `--grad-steps`, `--lr`, `--width`,
`--hidden`, `--eval-every`, `--eval-games`, `--device cpu|mps|cuda`,
`--resume checkpoints/latest.pt`.

Evaluate a checkpoint and build its report page:

```bash
uv run python examples/evaluate.py --games 500 --symmetric --json checkpoints/eval.json
```

```bash
uv run python -m game2048.report --log checkpoints/log.csv --eval checkpoints/eval.json --out checkpoints/report.html
```

`--symmetric` averages V over the eight board symmetries at play time, which
the **nn** agent in the UI also does. It is worth 16 points of 2048 rate
(57% to 74%).

The run in the results table: 46 minutes on an M3 Pro (MPS), 25 minutes at
learning rate 3e-4 and then 21 minutes resumed from the best checkpoint at
1e-4. 53.3M moves, 54,645 games.
