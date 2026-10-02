# Cool mode: placing your own tiles

In cool mode nothing is random. After every move the player picks an empty
cell and puts a 2 or a 4 there. That turns 2048 from a game of chance into a
puzzle with one question: how high can the score go?

This page is the full account of how the `snake` agent got to 3,931,768
points, 396 short of the ceiling. The short version is in the README.

## Where the ceiling is

Every point in 2048 comes from a merge: merging two tiles of value v scores
2v. So a tile's history fixes what it earned. A 4 built from two 2s earned 4,
an 8 built from 2s earned 16, and in general a tile 2^k built entirely from 2s
earned (k - 1) x 2^k. A 4 that was placed directly earned nothing, which is 4
points less than building it.

That gives the score of any finished game:

    score = (what the final board would have earned if built from 2s) - 4 x (number of 4s placed)

The largest tile a 4x4 board can hold is 131072. Making a tile needs two of
the tile below it at once, and the second of those needs the whole chain
beneath it on the board at the same time. For 131072 that is a 65536, then
32768 down to 4, plus one more 4 to start the cascade: 16 tiles on 16 cells.
A second 131072 would need 17.

So no final board beats one of each tile from 131072 down to 4, and built
from 2s that board is worth **3,932,164** points. That is the ceiling used
on this page. It is an upper bound and not quite reachable: the cascade above
only starts if its last tile is placed as a 4, and the last cell of the game
is filled by a placed tile too, so at least 8 of those points are out of
reach. Past reaching that board, the only thing left to optimise is how few
4s get placed.

## Search instead of learning

With no chance nodes, the search maximises over moves and over placements.

**Max-max tree** (`NTupleNet.best_choose`, `choose_value`, `play_choose`):
`depth` counts placement decisions, and `topk` placements are kept per node.

**Beam search** (`beam_plan`, `beam_choose`, `beam_value`, `play_beam`): keep
the `width` best boards, extend each by a move and a placement, repeat `depth`
times, then play the first steps of the best line. A board is ranked by its
rewards so far plus the best next move's reward and value. With unlimited
width this is exactly the max-max tree of the same depth, and a test checks
that. `stride` commits several steps of a plan before searching again, and
`spread` caps how many survivors may share a parent.

On tables trained for 5 minutes on the choose game, 6 games each:

| search | mean score | cost per move |
| --- | --- | --- |
| max-max tree, depth 3 | 599k | 9 ms |
| beam, width 16, depth 16 | 369k (stalls at 16384) | 16 ms |
| beam, width 32, depth 12 | 808k | 20 ms |

Width matters more than depth: a narrow beam fills up with variants of one
line.

A deterministic policy also makes the game a single line. Different start
tiles, and even a prefix of random spawns (`prefix=`), converge to the same
game within a few hundred moves. So two evaluation games are as good as
twelve, and small score differences between search settings are chaotic
rather than statistical.

## Dropping the learned tables

Tables trained on the choose game for 4 hours (`--choose`, see
[TRAINING.md](TRAINING.md)) reached 65536 and 1,528,428 points with the beam.
They were the weak link in the endgame, where a 16-tile chain has to be
collapsed in exact order.

The `snake` agent drops them. It runs the same beam on an untouched network,
whose value is 0 everywhere, so boards are ranked by their merges plus one
hand-written term: the **snake score**. Read the tiles along the path

     0  1  2  3
     7  6  5  4
     8  9 10 11
    15 14 13 12

weight the k-th by decay^k, and take the best of the 8 board symmetries. A
chain laid out along the path, biggest tile first, scores highest. That keeps
the board in a shape where each merge sets up the next one.

Settings: 128 lines, 16 steps ahead, 4 steps played per plan, snake weight 2.

| evaluator | beam | score | max tile | moves | time |
| --- | --- | --- | --- | --- | --- |
| best learned tables, no snake term | 32 x 12 | 1,528,428 | 65536 | 35,686 | 3 min |
| snake only, decay 0.5 | 32 x 12 | 785,284 | 32768 | 16,405 | 37 s |
| snake only, decay 0.5 | 64 x 16 | 1,614,376 | 65536 | 30,801 | 2.4 min |
| snake only, decay 0.5 | 128 x 16 | 1,704,108 | 65536 | 32,806 | 2.3 min |
| snake only, decay 0.75 | 128 x 16 | 3,670,436 | **131072** | 65,636 | 4.3 min |
| snake only, decay 0.9 | 128 x 16 | 3,670,068 | **131072** | 65,544 | 4.2 min |
| decay 0.9, 2s only (`tiles=1`) | 128 x 16 | 1,835,012 | 65536 | | |
| decay 0.9, 4s charged (`tiles=7`) | 128 x 16 | 1,742,508 | 65536 | 42,402 | 2.8 min |
| decay 0.9, 2s first (`tiles=12`) | 128 x 16 | **3,931,768** | **131072** | 130,969 | 4.1 min |

The decay was the first fix. At 0.5 only the head of the chain carries
weight, and the tail drifts out of order. At 0.75 or 0.9 every tile on the
path counts, and the beam reaches 131072.

## The last 262,000 points

The first games to reach 131072 scored about 3,670,000. They placed almost
nothing but 4s, and by the formula above that costs 4 points per tile, about
262,000 in total.

Three attempts to get those points back:

1. **2s only** (`tiles=1`). The beam builds a flawless chain from 65536 down
   to 2 and then stops. The next level needs 17 tiles on 16 cells unless one
   of them arrives as a 4.
2. **Charge each 4 its 4 points in the ranking** (`tiles=7`). Not enough on
   its own: a 4 still adds twice the tile mass per step, and this game died
   at 65536.
3. **2s first** (`tiles=12`). Plan with 2s only. If the best 2s-only line is
   dead at the end of the horizon, plan again with 4s allowed and play only
   that plan's first step, so the next plan tries 2s again.

The third one works. The game places 98 fours, in bursts of seven or eight
at the cascades where a full chain collapses, and ends with one of each tile
from 131072 down to 8 plus a 2 in the last cell. The fours cost 392 points
and the last cell another 4: 396 short of the ceiling, 99.99% of it. It takes
twice as many moves as the all-4s game, since every tile arrives as a 2, and
the same wall time, since a 2s-only beam has half the candidates.

What is left is the number of 4s per cascade, a few hundred points at most.

## With random tiles

The same heuristic can play the normal game. There the `snake` agent runs
depth-3 expectimax with the snake score added at the leaves
(`NTupleNet.leaf_snake`, weight 16, decay 0.9). Over 24 games it averages
66,199 points and reaches 2048 every time, 4096 in 62% of games and 8192 in
12%. Without the term the same search averages about 20,000. A bonus for
empty cells made it worse at every weight tried. The trained `ntuple` player
is five times stronger.

## In Python

```python
from game2048 import ntuple as nt

net = nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False)   # empty tables: value 0 everywhere
net.snake_decay = 0.9
stats = net.play_beam(games=1, width=128, depth=16, stride=4, snake=2.0, tiles=12)
print(stats["scores"], stats["max_tiles"], stats["moves"])
```

`tiles` is a bit mask: 1 allows 2s, 2 allows 4s, 4 charges each placed 4 its
4 points in the ranking, and 8 turns on 2s first.
