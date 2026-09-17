# Handoff: 2048 project state (updated 2026-09-16, after the cool-mode trainer)

Everything needed to pick this up cold. Read this, then `README.md`.

## What exists and works

- **Game + web UI**: `game2048/core.py` (rules), `game2048/server.py` (FastAPI),
  `static/` (board UI). Run with `uv run uvicorn game2048.server:app --port 8048`
  (port 8000 is taken by Docker). Tile modes: random, kind, best, evil, and
  **choose** ("cool mode": after each move the player clicks an empty cell and
  picks 2 or 4; the API is `POST /api/place {row, col, value}`; agents do
  move + placement in one `POST /api/agent/step`).
- **Agents in the UI dropdown**: random, greedy, nn (CNN, weak), ntuple (the
  strong one), and ntuple-cool (503 until its weights exist, see below).
- **CNN value net** (`game2048/nn.py`, `train.py`): 74% reach 2048. Historical.
- **n-tuple engine** (`game2048/native/ntuple.c`, wrapper `game2048/ntuple.py`,
  trainer `game2048/ntuple_train.py`): bitboards, TD/TC learning, multi-stage
  tables, expectimax, and the choose-mode max-max search (`best_choose`,
  `choose_value`, `play_choose`). Compiles itself with `cc` on first import.
- **Tests**: `uv run pytest` -> 173 pass as of this handoff.
- **Reports** (artifacts): CNN https://claude.ai/artifact/9nB633R3Nz1UhUv8saPvLs,
  n-tuple https://claude.ai/artifact/RqGV3yvYbAHagKPmot4rRc (8-pattern run).

## Trained weights

| File | What | 100-game result (seed 2024) |
| --- | --- | --- |
| `checkpoints/ntuple/weights.bin` | **deployed** 8x6 n-tuple, 4 h, 4.0M games | depth 3: mean 351,671, 16384 98%, 32768 1%, best 621,048 |
| `checkpoints/ntuple/weights_4x6.bin` | 4x6 n-tuple, 2 h, 4.5M games | depth 3: mean 345,675, 16384 97%, 32768 0%, ceiling ~385k |
| `checkpoints/ntuple_8x6/latest.bin` | same 8x6 with TC learning state (1.5 GB), resumable | |
| `checkpoints/ntuple_stages/` | 3-stage attempt seeded from 4x6; **worse** (313k at depth 2) | |
| `checkpoints/best.pt` | CNN | 74% reach 2048 |

Evaluate any weights: `uv run python examples/evaluate_ntuple.py --weights <file> --games 100 --depths 2,3 --threads 12 --seed 2024`
(depth 3 on 100 games takes ~16 min alone on 12 threads; never run two heavy jobs at once, see Machine).

## Cool-mode value function: trainer built 2026-09-16, training NOT started

Finding: with the player placing tiles, the deployed tables still cap at 16384 /
~357k, and depth 2 scores below depth 1. Cause: those tables were trained on
random-spawn games and never saw a 32768 on the board, so every entry that
includes a 32768 is 0 and the search avoids creating one. Cool mode needs its
own tables trained by self-play where the agent places the tiles.

What exists now (all tested; `uv run pytest` -> 173 pass):

- C `nt_train_choose(net, games, threads, alpha, alpha_plain, seed, depth, topk,
  max_moves, scores, maxtiles, moves)` in `ntuple.c`: TD(0) on afterstates, the
  policy is the choose search at `depth` (1) on a context created without a
  transposition table (`ctx_create_tt(net, 0.0, 0)`; `choose_av` guards `c->tt`).
  Terminal update `0 - V(prev)` when no move exists; a game cut at `max_moves`
  gets no terminal update.
- `NTupleNet.train_choose(games, threads, alpha, seed, max_moves, alpha_plain,
  depth=1, topk=4)` -> per-game stats like `train`.
- CLI: `--choose` and `--max-moves` (default 200000). `state.json` records
  `choose` and `max_moves`; resuming a `--choose` run keeps choose mode; `eval.csv`
  has a `mode` column and evaluations use `play_choose`. Appending to an older
  csv keeps that file's columns.
- Agent `ntuple-cool` (env `GAME2048_NTUPLE_COOL`, else
  `checkpoints/ntuple_choose/weights.bin`), listed by `/api/agents`; the server
  answers 503 until the weights exist. The UI dropdown is built from that list.

**Lesson: never cap the games.** With a 2000-move cap in the tests, once ~25% of
games hit the cap the late boards were never grounded, values inflated (start
board valued 5e4 while games actually scored 4k) and the policy collapsed to
dying at move 250. Uncapped, learning is stable: a throwaway 4x6 net on 8
threads reached mean 148k with a 16384 tile after 9600 games (2.3 min). Games
in the 4-bit engine always end (mass is bounded), so the cap is only a guard.

Launch (detached, one heavy job at a time; ~1.5 GB with `--patterns 8x6`):

```
uv run python -m game2048.ntuple_train --choose --patterns 8x6 --hours 3 --threads 12 \
  --lock-memory --out checkpoints/ntuple_choose --chunk 1000 --save-every-min 20 \
  --eval-every-min 30 --eval-games 12 --eval-depth 2 --eval-at-end --seed 5
```

Choose-mode games are long (thousands of moves, ~480 evaluations each), so use
`--chunk 1000`, not 20000. Evaluate afterwards with
`uv run python examples/evaluate_ntuple.py --weights <file> --choose --games 2
--depths 3 --beam 32:12,64:12:4 --threads 4` (beams are width:depth[:spread[:stride]];
`--json` writes the usual evals file with `meta.mode = "choose"`). Then copy `weights.bin` where the ntuple-cool agent looks (already the
default `--out`), restart the server, update README results and the memory note.

**5-minute trial run (2026-09-16 14:10, seed 5, 8x6, 12 threads):** 10,000 games,
35M moves, ~110k moves/s; training (depth-1) mean rose 34k -> 200k with 16384 in
~30% of games. Weights saved in `checkpoints/ntuple_choose/` (latest.bin 1.6 GB,
weights.bin 512 MB), so `ntuple-cool` is live in the UI. 12-game evaluation with
`play_choose`, seed 2024 (games are near-deterministic in cool mode):

| depth | mean | max tile | moves/game | time |
| --- | --- | --- | --- | --- |
| 1 | 163,508 | 8192 | 4,250 | <1 s |
| 2 | 165,512 | 8192 | 4,491 | 4 s |
| 3 | 599,177 | 32768 in 12/12 | 14,743 | 93 s |

Use `--eval-depth 3` for future runs (depth 2 is misleading here). Resume the
trial with `--resume checkpoints/ntuple_choose/latest.bin --out checkpoints/ntuple_choose`
(choose mode is remembered in state.json).

**Beam search (built 2026-09-16 afternoon, tested):** `nt_beam_plan/choose/value`
and `nt_play_beam` in `ntuple.c`; Python `NTupleNet.beam_plan(bits, width, depth,
spread)`, `beam_choose`, `beam_value`, `play_beam(games, width, depth, threads,
seed, max_moves, stride, spread)`; `NTupleAgent(beam_width, beam_depth,
beam_stride, beam_spread)` caches its plan between `choose` calls (invalidated when
the board differs from the plan's next board). Unlimited width == full max-max
tree (reference test). On the 5-minute tables (frozen copy), 6 games, seed 2024,
timings under training contention: tree depth 3 599k / 9 ms per move; beam w8 d8
274k; w16 d8 370k; w16 d16 369k (16 ms); **w32 d12 808k, 32768 in 6/6, 20 ms**.
w64 d16 719k (59 ms). Width beats depth up to a point. `ntuple-cool` now plays beam w32 d12. Round 2 (spread 2/4,
stride 4, w64) results go in `scratchpad/beam_compare.log` and this doc.

**Exploration (built, tested, not yet evaluated):** `train_choose(..., explore=p)`
/ CLI `--explore p` places a uniformly random tile on a fraction `p` of steps so
training games leave the single line a deterministic policy replays (on-policy
TD, the noisy policy is what gets valued). Queued experiment
(`scratchpad/explore_experiment.sh`, runs after the 4-hour run exits): continue
the 4-hour tables for 1 h with `--explore 0.02` into `checkpoints/ntuple_choose_explore/`
and 1 h without into `checkpoints/ntuple_choose_ctrl/`, each ending with a 2-game
depth-3 eval in its `train.log`/`eval.csv`. Compare the two, then run beam w32 d12
on both `weights.bin` files; if exploration wins, resume the main tables with it.

**Round 2 on the 5-minute tables (2 games each, canonical line, 2 threads under
contention):** tree d3 599k (8 ms/move); beam w32 d12 808k (18 ms); w32 d12
spread4 612k; w32 d12 spread2 959k (two 32768s); w32 d12 stride4 799k (4 ms);
w64 d8 375k (dies at 16384); **w64 d12 spread4 1,162k** (two 32768s + 16384,
32.6k moves, 29 ms); w128 d12 spread4 stride4 788k (15 ms). Robust reading:
depth >= 12 is required (every d8 config dies at 16384), width >= 32 helps, stride
4 costs almost nothing, spread is a coin flip per snapshot. The same script
(`scratchpad/round2.py <weights> <tag>`) is queued on the finished tables
(`scratchpad/final_compare.sh`, output tagged `[final]` in `beam_compare.log`) with
extra w64 variants; pick the UI default from a config that does well on both.

**Evaluation is one game.** Cool mode under a deterministic policy funnels every
start into the same line within a few hundred moves: 12 games score within a few
points of each other, and a random-spawn prefix (`prefix=` on `play_choose`/
`play_beam`, kind spawns so the prefix does not kill the game; CLI `--eval-prefix`)
still converges (std ~650 on 164k with prefix 200; prefix 500+ starts killing
games). So evaluate with `games=2`, and treat differences between search configs
as chaotic (w64 d16 scored 719k where w32 d12 scored 808k on the same tables).
Compare configs across several weight snapshots rather than across games.

**Bug fixed on the way:** `to_bits` guessed values-vs-exponents by `max > 15`, so
early boards of 2s/4s/8s were read as exponents; agents and the evaluator now use
`tiles_to_bits` (test `test_agent_sees_the_real_board_early_in_a_game`).

Fallback if learning stalls late: a hand-written snake/monotonicity heuristic
as the leaf evaluator for the max-max search.

## Tiles above 32768 (engine change, 2026-09-16 evening)

The C engine now uses 128-bit boards with five bits per cell (`board_t` =
`__uint128_t`, `cell()`/`put()` helpers, `MAX_EXP 17` = 131072, the 4x4 maximum)
and radix-18 table indexing (`RADIX 18`: 6-cell patterns have 18^6 = 34M entries,
136 MB each; an 8x6 net is 1.09 GB of weights, 3.3 GB with TC state). Boards
cross the C API as two uint64 halves (`_lohi` in `ntuple.py`; Python-side helpers
`nt.cell`, `nt.with_cell`, `nt.CELL_BITS`). File format `NTUPLE06`; every older
file (radix 16) is re-indexed on load (`remap_old_index`), verified against a
fixture made with the old library (`tests/fixtures/ntuple05_*.bin`,
`tests/test_ntuple_bigtiles.py`). The row LUT is 2^20 entries (16 MB). Training
jobs started before the change keep running on the old library (the dylib is
replaced atomically) and still cap at 32768; the tables they save convert on load.
`vec_env.py` (the numpy engine behind the CNN) still caps at 32768.

Next training run should resume the 4-hour cool-mode tables with the new engine so
the value function learns 65536 and 131072:
`uv run python -m game2048.ntuple_train --resume checkpoints/ntuple_choose/latest.bin --out checkpoints/ntuple_big --hours 3 --threads 12 --lock-memory --chunk 1000 --save-every-min 20 --eval-every-min 30 --eval-games 2 --eval-depth 3 --eval-at-end --seed 8`
(3.3 GB pinned; run it alone).

## Snake-order bonus, exploration verdict, and the big-engine run (2026-09-16 night)

- **Snake bonus**: `nt.snake_score(bits)` = tiles read along the snake path
  (0 1 2 3 / 7 6 5 4 / 8 9 10 11 / 15 14 13 12) weighted 0.5^k, best of the 8
  symmetries; `beam_plan/beam_choose/beam_value/play_beam(..., snake=w)` add
  `w * snake_score` to every beam leaf; `NTupleAgent(beam_snake=w)`. Motivation:
  the user watched the endgame die with the chain scattered. Sweep of w on the
  4-hour tables (canonical game, beam w32 d12 stride 4) is in
  `scratchpad/snake_sweep.log`; w = 0 scored 1,299,016 with a 65536 on the new
  engine (1,216,968 with the old cap).
- **Exploration**: 1 h continuations of the same 2h20m snapshot, 4 threads each,
  final 2-game depth-3 eval: explore 0.02 -> 802,988; control -> 751,148. One line
  each, so chaotic, but exploration is on for the next run.
- **Queued** (`scratchpad/launch_big.sh`, starts when the 4-hour run exits): resume
  the 4-hour tables on the new engine into `checkpoints/ntuple_big/` for 3 h with
  `--explore 0.02` (3.3 GB pinned), evals every 30 min at depth 3, 2 games.

## Machine constraints (important)

- 19 GB RAM, but Docker's VMs hold ~9 GB; swap ran 4-10 GB. Pinned tables above
  ~1.6 GB thrashed. Run one heavy job at a time. `--lock-memory` pins tables.
- 12 cores: 6 performance + 6 efficiency; the trainer uses an atomic work counter.
- Training budgets use a monotonic clock, so laptop sleep pauses a run instead of
  consuming its budget; detached runs survive the Claude session ending.
- `checkpoints/*/train.log` has one line per 20k-game chunk plus `[eval ...]`
  lines; `state.json` has counters for `--resume`.

## Open decision (user's)

Beating the published record (Jaskowski 2018: ~609k mean, 32768 in ~70%) needs
Docker quit for memory, an 8x6 3-stage net with carousel shaping (train late
stages from saved late-game positions), and 12-24 h. Not started.

## Restart checklist

1. `uv run pytest` (expect 173 pass).
2. Launch the choose-mode training above, detached (subprocess with
   `start_new_session=True`, stdout to `checkpoints/ntuple_choose/train.log`);
   monitor the log.
3. `uv run uvicorn game2048.server:app --port 8048` for the UI (or the "2048"
   entry in `.claude/launch.json`); verify cool mode by playing a move, clicking
   a cell, picking 2/4. `ntuple-cool` works in the dropdown once weights exist.
4. When choose weights exist: play_choose at depth 2-3, report max tile / score,
   update README results and the n-tuple memory note.
