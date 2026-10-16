# Training log

Every training run behind this repository, in order: what was trained, on what,
for how long, and what came of it. All of it is self-play. No player here has
seen a game it did not play itself.

Last updated 2026-10-16, 11:40. No training has run since run 26; the work
since then has been search and endgame tables, which are not training; the
tables' build time is listed under Endgame tables.

## Totals so far

| Machine | Wall-clock | CPU | Self-play moves |
| --- | --- | --- | --- |
| MacBook (M3 Pro, 12 cores) | 16.2 h | 168 thread-hours, plus 71 min on its GPU | 116 billion |
| ThinkPad (i5-1240P, 16 threads) | 72.3 h | 1,153 thread-hours | 1,130 billion |
| Google Colab, CPU runtimes | 73.4 h | 585 vCPU-hours | 793 billion |
| Google Colab, GPU runtimes | 1.2 h | 1.2 GPU-hours | 10 billion |
| **All** | **164 h** | **1,907 CPU thread-hours** | **about 2,050 billion** |

A thread-hour is one hardware thread busy for an hour. The threads are not
equal: the ThinkPad's are held to about 20 W by its firmware and a Colab vCPU
is half of an older Xeon core, so the column measures effort, not work done.
Moves are the better measure of training.

Runs 7, 8 and 9 were resumed from run 6 and carried its clock forward, so the
`elapsed_s` fields of the checkpoints add up to more than the wall-clock here.

## The machines

- **MacBook**, Apple M3 Pro, 6 performance and 6 efficiency cores, 18 GB. The
  first machine, and the fastest. It is also the everyday laptop, so its runs
  are hours long at most.
- **ThinkPad** X1 Carbon Gen 10, Intel i5-1240P, 16 threads, 16 GB. Slower, but
  it can stay plugged in with nothing else to do, so the runs that take days
  happen here. Its firmware limits the processor to about 20 W under sustained
  load: 5.1M moves a second on eight patterns, after the engine was reworked
  for it, against 2.6M when it started.
- **Google Colab**, rented by the hour. The high-RAM CPU runtime (8 vCPUs,
  50 GB) runs the same C engine at 3.5M moves a second and is used for
  rehearsals and evaluations that should not take time from the ThinkPad.
  Sessions end after 12 to 24 hours, so a job there is watched from the
  ThinkPad, which keeps its files and starts it again on a new machine. GPU
  runtimes were tried once (run 16).

## September 2026: the first players (MacBook)

### 1. CNN value network, the baseline (15 September)

- **What:** a small convolutional network (128 channels, 512 hidden units),
  afterstate TD(0), 512 boards in parallel with a replay buffer of a million
  positions.
- **On:** PyTorch on the laptop's GPU.
- **Time:** 71 minutes. 50 at learning rate 3e-4, of which the first 25 were
  kept (the evaluations drifted down after that), then 21 at 1e-4 from the best
  checkpoint. The kept line is 46 minutes, 53.3M moves, 54,645 games.
- **Result:** 500 games: mean 29,869 with 2048 in 73.6% of them, averaging over
  the 8 board symmetries; 23,392 and 57.2% without. It stalled there. Kept as
  the `nn` player.

### 2. N-tuple network, 4 patterns of 6 cells (15 September)

- **What:** lookup tables, TD(0) with temporal coherence learning, in C.
- **On:** 12 threads, 0.8 GB.
- **Time:** 2.0 hours. 4.5M games, 39.1B moves, 6.5M moves a second.
- **Result:** 100 games: 241,803 with no search, 345,675 at depth 3, 16384 in
  97% and 32768 in none. It tops out near 385,000, where a game ends if it
  never merges two 16384s.

### 3. The same with three stages (15 September, dropped)

- **What:** run 2's tables as the first stage, with separate tables for boards
  past a tile mass of 16384 and of 24576.
- **On:** 12 threads, 2.3 GB.
- **Time:** 2.05 hours. 3.54M games, 32.8B moves.
- **Result:** worse than run 2. Its depth-2 evaluations slid from 354k to 299k,
  and a 100-game test gave 313,338 against run 2's 343,454. Three variants were
  tried briefly and abandoned (plain TD at rate 0.1, plain TD on the later
  stages only, and four stages, which ran the machine out of memory); their
  times were not recorded.

### 4. N-tuple network, 8 patterns of 6 cells (15 September)

- **What:** run 2's method with twice the tables, one stage.
- **On:** 12 threads, 1.5 GB.
- **Time:** 4.0 hours. 4.0M games, 36.4B moves, 2.7M moves a second.
- **Result:** the `ntuple` player in the README. 100 games: 276,699 with no
  search, 342,382 at depth 2, 351,671 at depth 3 (best game 621,048), 16384 in
  98% and 32768 in 1%.

### 5. Cool-mode smoke tests (16 September)

- **What:** two short runs on the game where the player places every tile.
- **Time:** 2.3 minutes (4 patterns) and 5 minutes (8 patterns).
- **Result:** games with no move cap learn stably. The 5-minute tables were
  used to compare searches (tree at depth 3: 599k; beam of width 32, depth 12:
  808k). Both overwritten by run 6.

### 6. Cool-mode tables, 8 patterns (16 September)

- **What:** TD(0) on the placing game, a depth-1 search over move and
  placement as the policy, no exploration.
- **On:** 12 threads, 1.5 GB.
- **Time:** 4.1 hours. 104,000 games, 1.5B moves: each game runs for thousands
  of moves with a search at every one.
- **Result:** the `ntuple-cool` player. With the beam: 1,528,428 points and a
  65536 tile, the best result from learned tables. The untrained snake search
  passed it that evening and now scores 3,932,100.

### 7 and 8. Exploration, with and without (16 September)

- **What:** two branches from run 6's two-hour snapshot, one unchanged, one
  placing a random tile on 2% of steps.
- **On:** 4 threads each.
- **Time:** 0.9 and 1.0 hours, 4,000 and 6,000 more games.
- **Result:** 751,148 without and 802,988 with, one game each on a chaotic
  line, so weak evidence. Exploration was used for run 9.

### 9. Cool-mode tables on the wider engine (16 September, stopped)

- **What:** run 6's tables converted to the engine that holds tiles up to
  131072, with 2% exploration and 30% of games restarted from late boards.
- **On:** 12 threads, 3.0 GB, the machine's limit.
- **Time:** 0.5 of a planned 3 hours, 5,000 more games.
- **Result:** stopped by hand: the snake search solved cool mode without tables
  that evening.

### 10. Quick-start check for the README (2 October)

- **What:** fresh 4-pattern tables on the current engine, on a busy machine.
- **Time:** 10 minutes. 140,000 games, 563M moves.
- **Result:** 48 games at depth 2: mean 201,797, 8192 in 96%. This is the
  "ten minutes" figure in the README. The files were not kept.

## October 2026: after the published record

The goal changed here: reproduce the best published learned player (Guei, Chen
and Wu, 2022) and then pass it. Their recipe is in [TRAINING.md](TRAINING.md).

### 11. Speed benchmarks for the training-loop rewrite (9 October, MacBook)

- **What:** timed chunks of plain-TD self-play on run 4's tables, before and
  after rewriting the inner loop.
- **Time:** about 1.5 minutes in all, 0.2B moves. Nothing written to disk.
- **Result:** eight patterns went from 2.26M to 9.20M moves a second on 12
  threads, with identical scores single-threaded.

### 12. Optimistic TD, first attempt (9 October, MacBook, stopped)

- **What:** the first step of the record recipe: Matsuzaki's eight patterns,
  every board valued at 320,000 to begin with, plain TD(0) at a fixed rate of
  0.1.
- **On:** 12 threads, 1.1 GB.
- **Time:** 11.6 minutes of a planned 27 hours. 1.5M games, 6.0B moves.
- **Result:** stopped before its first save, because a run of that length
  belongs on a machine with nothing else to do. Training games averaged
  124,736 at the end. Restarted from zero as run 14.

### 13. Benchmarks and engine work on the ThinkPad (10 October)

- **What:** a five-minute throwaway network, then timed chunks at 4 to 16
  threads, under two power profiles, and before and after each change to the
  engine (huge pages, vector indexing, transpose, spawn).
- **Time:** about half an hour in all, about 4B moves. Nothing kept.
- **Result:** the firmware's 20 W limit, not the thread count, sets the speed.
  The changes took eight patterns from 2.6M to 5.1M moves a second.

### 14. Optimistic TD, stage 1 (10 to 12 October, ThinkPad)

- **What:** run 12 again, for the full budget: about 0.9e12 moves of plain TD,
  to be followed by a tenth as much with temporal coherence.
- **On:** 16 threads, 1.1 GB. Stopped and resumed twice in its first three
  hours to pick up faster builds of the engine.
- **Time:** 52.5 hours, ending 08:14 on 12 October. 95.0M games, 925.6B moves,
  at about 5.1M moves a second throughout.
- **Outcome:** training games average about 276,000 over the last five million,
  with 16384 in 67% and 32768 in 6.0%. The 32768 rate stayed under half a
  percent for the first 17 hours, passed 1% at 33.1M games and was still rising
  slowly at the end (5.0% at 50M games, 5.6% at 70M). Alone at depth 2, the
  hourly 48-game checks of the last day average about 395,000, with 32768 in
  about a third of games. Copies of the weights were kept every two and a half
  hours. The temporal coherence phase has not been run yet.

### 15. Speed checks on Colab's CPU runtimes (10 October)

- **What:** the C engine training from scratch, to see what the runtimes do.
- **On:** the high-RAM runtime (8 vCPUs) for 7 minutes and the standard one
  (2 vCPUs) for 5.
- **Result:** 3.54M and 0.99M moves a second, about 2B moves in all. The
  high-RAM runtime does the same work for slightly less and in under a third of
  the time.

### 16. A GPU version of the trainer (10 October, Colab, experiment)

- **What:** the same learning rule in PyTorch, with thousands of games stepped
  together on a GPU. Not in the repository.
- **On:** T4, L4 and A100 runtimes, about 1.2 hours in all, about 10B moves.
- **Result:** summing the updates of every game that touches a table entry in
  one step multiplies the learning rate, and the first version learned nothing
  in a million games. With each entry moved once per step it followed the C
  engine's learning curve to a million games at batches of 16,384 and 262,144.
  Speeds: 2.1M to 2.5M moves a second on a T4, 5.8M on an L4, 17M to 18M on an
  A100. Per move the CPU runtime costs a quarter as much, and this version has
  not been checked past a million games, so the runs stay on the C engine.

### 17. Test of the stop-and-resume runner (10 October, Colab)

- **What:** a 4-pattern network on the standard runtime, with its session
  killed on purpose half-way.
- **Time:** about 7 minutes, 0.6B moves.
- **Result:** the watcher started a new machine, sent the saved checkpoint back
  and the run finished from where it had been.

### 18. Rehearsal of the switch to temporal coherence (10 October, Colab)

- **What:** a copy of run 14 at 20% of its budget, trained for an hour with
  temporal coherence at rate 1.0.
- **On:** 8 vCPUs, 3.3 GB.
- **Time:** 1.0 hour. 700,000 games, 8.05B moves.
- **Result:** nothing unstable. Depth 2 over the same 200 games: 307,447 before
  and 346,172 after. A rehearsal only: run 14 was not touched.

### 19. Rehearsal of stage 2 (10 October, Colab)

- **What:** the whole second-stage pipeline, with the network released with
  TDL2048 (Guei's own single-stage network) standing in for our stage 1, which
  is not finished: 200,000 boards collected where it first makes a 16384, a
  second network trained from those boards with its own optimistic start, the
  two joined, and the result evaluated.
- **On:** 8 vCPUs, 2.2 GB.
- **Time:** 6.0 hours of training. 17.1M games from the collected boards, 75.0B
  moves.
- **Result:** every step worked. The joined player scored 311,914 with no
  search over 2,000 games (32768 in 8.6%) and 431,822 at depth 2 with tile
  downgrading over 1,000 (32768 in 31.4%). That is below the stand-in alone
  (375,426 and 537,396): the second network had a sixth of its budget and no
  temporal coherence phase, and was still climbing when it stopped. Half of
  this player is someone else's network, so it is a test of the pipeline and
  nothing more.

### 20. Stage 2, started early (11 to 12 October, Colab)

- **What:** the second network, for boards that hold a 16384. It reads stage 1
  only through the boards where stage 1 hands over, and those are nearly empty
  boards with one 16384 whatever stage 1's strength, so it does not have to
  wait: a million hand-over boards are collected from run 14 as it stood at
  40.05M games, and the second network trains from them with its own optimistic
  start, plain TD at 0.1. When run 14 is finished its hand-over boards will be
  compared with these, and the run moved onto them if they differ.
- **On:** the high-RAM CPU runtime, 8 vCPUs, 2.2 GB, restarted by the watcher
  on the ThinkPad whenever a session ends.
- **Budget:** 440B moves, about 100 million games from these boards, which is
  the published length for a stage. Started 01:06 on 11 October.
- **Time:** 35.7 hours, ending 12:45 on 12 October: 78.65M games, 440.2B moves,
  at 3.4M to 4.0M moves a second. The games came out longer than estimated, so
  the move budget bought fewer of them. Colab ended the first session after 24
  hours; the watcher started a new machine and resumed from the last copy it
  held, losing about 50 minutes.
- **Outcome:** training games, which start with a 16384 on the board, reach a
  32768 in 15%. Run 14's finished network hands over somewhat different boards
  from the ones collected at 40.05M games (the 16384 in a corner on 84% of them
  against 89%), so the last phase of this network will train on boards from the
  finished stage 1.

### 21. Stage 1, temporal coherence phase (12 October, ThinkPad)

- **What:** the last tenth of stage 1's training: run 14's weights, continued
  with temporal coherence learning at a step of 1.0 for 9.5M games.
- **On:** 16 threads, 3.3 GB (the phase keeps two more tables beside each
  weight table). About 2.9M moves a second.
- **Time:** 12.7 hours, 08:44 to 21:30. 130.4B moves.
- **Outcome:** the gain came at once and then stopped. Training games averaged
  323,767 over the first 50,000 against about 276,000 at the end of run 14,
  about 370,000 from the third million on, with 16384 in 85% and 32768 in 24%.
  Alone at depth 2 over the same 300 games, the weights at 1.8M, 4.7M and 6.4M
  games of the phase score 475,415, 477,034 and 467,637 (each within about
  14,000), with 32768 in a little over half the games; before the phase the
  hourly checks averaged about 395,000.

### 22. Stage 2, temporal coherence phase (12 to 13 October, ThinkPad)

- **What:** the last tenth of stage 2's training: run 20's weights, continued
  with temporal coherence learning for 7.9M games, from a million hand-over
  boards collected afresh from the finished stage 1 (the 16384 in a corner on
  87% of them).
- **On:** 16 threads, 4.4 GB. About 3.0M moves a second.
- **Time:** 6.6 hours, 21:53 to 04:27. 69.6B moves.
- **Outcome:** training games, counted from the hand-over, average about
  241,000 against about 156,000 at the end of run 20, and reach a 32768 in 36%
  against 15%. Most of that came in the first hour. This is the last training
  run of the two-stage player. The two networks joined average 565,703 over
  1,000 games at depth 2 with tile downgrading and 589,335 at depth 3; the
  README has the full table and the weights are attached to the v2.0 release.

### 23. Engine speed work and a one-hour pilot (13 October, ThinkPad and Colab)

- **What:** short training runs used only to time the engine (up to a few
  minutes each, on the finished networks, thrown away afterwards), and one
  hour of stage 2's finished network trained further with the expected-value
  target, to see whether it changed anything quickly.
- **On:** the ThinkPad, 16 threads; one Colab CPU machine for about an hour of
  timing.
- **Time:** the pilot ran 61 minutes: 510,000 games, 4.7B moves.
- **Outcome:** the engine changes that came out of the timing (faster search,
  faster temporal coherence training, results unchanged) are in the code. The
  pilot changed nothing measurable over 1,000 games, and was not kept.

### 24. Stage 2 with the expected-value target (13 to 14 October, Colab)

- **What:** run 20's weights (stage 2 before its temporal coherence phase),
  trained further toward the average over every tile that can appear instead
  of the tile that did, at the same fixed learning rate, from the same million
  hand-over boards. Budget 70B moves.
- **On:** one Colab high-RAM CPU machine (8 vCPUs). About 1.2M moves a second:
  this target costs about five times as much per move.
- **Time:** 19:20 on 13 October to 11:53 on 14 October, 15.9 hours.
  8.56M games, 70.1B moves.
- **Outcome:** training games, counted from the hand-over, averaged about
  222,000 at the end and reached a 32768 in 31%, against about 156,000 and
  15% at the end of run 20 (the same weights trained on with the usual
  target) and 241,000 and 36% after run 22's temporal coherence phase. It gets
  its own temporal coherence phase next (run 26) before it is played against
  the v2 player.

### 25. Stage 3, for boards holding a 32768 and a 16384 (13 to 14 October, Colab)

- **What:** a third network, new, trained only from boards that already hold
  a 32768 and a 16384: 300,000 such boards collected from games of the v2
  player without search. Optimistic start at 190,000 (below the 320,000 of
  the first two stages, since less is left to win from these boards), fixed
  learning rate. Budget 190B moves.
- **On:** one Colab high-RAM CPU machine. About 3.9M moves a second.
- **Time:** 19:16 on 13 October to 10:00 on 14 October, about 12.6 hours of
  training after two hours collecting the boards. 63.5M games, 190B moves.
- **Outcome:** training games from those boards averaged about 74,000 at the
  end, up from 47,000, and flat for the last six hours. From real boards of
  that kind it played worse than the two-stage player does with tile
  downgrading (which plays such boards as their twins with half the tiles), and
  it was not kept. A network trained this way would need far more training to
  catch up with the 78M games stage 2 already had.

### 26. Temporal coherence phase for run 24 (14 October, Colab)

- **What:** run 24's weights, continued with temporal coherence learning and
  the expected-value target for 7B moves (a tenth of run 24), from the same
  hand-over boards. Then joined to the finished stage 1 and played over 1,000
  games at depth 2 with tile downgrading, the same test that gave v2 565,703.
- **On:** one Colab high-RAM CPU machine (8 vCPUs).
- **Time:** 12:29 to 14:48 on 14 October, after uploading its 3.3 GB of
  inputs; 105 minutes of training. 780,000 games, 7.1B moves.
- **Outcome:** training games averaged about 244,000 and reached a 32768 in
  37%, against 241,000 and 36% after run 22. Over the same 1,000 games at
  depth 2 with tile downgrading the joined player averaged 569,924 against
  v2's 565,703: 4,221 more a game, well inside the noise (plus or minus
  14,555), with a 32768 in 60.0% of games against 58.6%. Over 10,000 games
  on the same seeds (depth 2, tile downgrading, the 5-locked endgame tables)
  it averaged 565,047 against v2's 560,276: 4,770 more a game, plus or minus
  4,674, so just at the edge of the noise, with a 32768 in 58.1% against 57.7%
  and a 65536 in 0.16% against 0.20%. Too small a gain to switch; v2 stays the
  player.

## Endgame tables

Late in a game the player looks moves up in tables instead of trusting the
network alone. Each table holds every arrangement of the small tiles around a
fixed set of large ones, with the exact chance of reaching a goal from it and
the move that gets the best chance. Nothing in them is learned: they are solved
backward from the goal, once, and read from disk during play. They are not
training, but they are built ahead of time and the player needs them, so their
cost is listed here. All were built on the ThinkPad, 8 to 12 threads.

| Tables | Shape of the large tiles | Built | Positions | On disk | Build time |
| --- | --- | --- | --- | --- | --- |
| Snake ladder | five large tiles along the edge in a snake | 14 October | 18.2 billion | 36.3 GB | 32 min |
| Block ladder | five large tiles packed in the corner | 15 October | 18.2 billion | 36.3 GB | 35 min |
| Lift, one tile | the snake with one large tile moved off the path | 15 October | 4.6 billion | 9.2 GB | 41 min |
| Lift, two tiles | the snake with two large tiles moved off the path | 15 October | 1.0 billion | 1.9 GB | 9 min |
| Corner block, 2 by 3 | six large tiles in a 2 by 3 corner block, one table per smallest tile | 16 October | 2.9 billion | 5.8 GB | 19 min |
| Block, 3 and 2 | five large tiles, three on the edge and two below | 16 October | 7.1 billion | 14.2 GB | 31 min |
| **All six** | | | **52 billion** | **104 GB** | **2.8 h** |

On 16 October the four ladder sets (78 GB) were deleted to free disk: the
player as it stands, with the heuristic search and the two block sets, scores
the same without them. Built again from the same builder they come out
identical, so they can return when a player needs them.

Not kept: the snake ladder was deleted once to free disk and built again
(36 min), and a set of tables that let one large tile move around was built
(17 min, 8 GB) and dropped when it played no better. Small trial tables built
in seconds or minutes while the builder was written are not listed.

## Not training

- The `snake` agent has no trained weights: it is a search with one
  hand-written rule. Its cool-mode games, and the Lean proof that 3,932,100 is
  the most any game can score, involve no training.
- Evaluations (playing games with finished weights, with or without search)
  are not counted here, nor is collecting boards for a later stage.
- The test suite trains small throwaway tables for seconds at a time.
- The endgame tables above are solved, not learned, so their build time is not
  in the training totals.
