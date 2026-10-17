"""The rules of 2048 written out from scratch, with no code shared with the engine, to check a game trace.

A trace is a list of boards (4 x 4 grids of exponents: 0 empty, 1 for a 2, 2 for a 4, ...) and the move made on each
(0 up, 1 right, 2 down, 3 left; -1 on the last board). check_trace says what breaks the rules, or None:
  - the game starts with two tiles, each a 2 or a 4, and nothing else;
  - each move changes the board; tiles slide toward the move's side, and two equal neighbours (after the gaps close)
    merge once into one tile of twice the value, nearest the side first; a merged tile does not merge again that move;
  - after each move exactly one new tile, a 2 or a 4, appears in a cell the move left empty;
  - the points (the value of every tile made by a merge) add up to the claimed score;
  - the last board has no move left.

    python examples/rules2048.py <trace.npz> [score]
"""
from __future__ import annotations

import sys


def slide_row_left(row: list[int]) -> tuple[list[int], int]:
    """One row moved left: (new row, points). Exponents, 0 = empty."""
    tiles = [x for x in row if x]
    out, points, i = [], 0, 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            out.append(tiles[i] + 1)
            points += 2 ** (tiles[i] + 1)
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    return out + [0] * (4 - len(out)), points


def apply_move(board: list[list[int]], move: int) -> tuple[list[list[int]], int]:
    """The board after a move (0 up, 1 right, 2 down, 3 left) before the new tile, and the points."""
    if move not in (0, 1, 2, 3):
        raise ValueError(f"no move {move}")
    lines = []
    for k in range(4):                      # each line read from the side the tiles move toward
        if move == 3:
            lines.append([board[k][c] for c in range(4)])
        elif move == 1:
            lines.append([board[k][c] for c in range(3, -1, -1)])
        elif move == 0:
            lines.append([board[r][k] for r in range(4)])
        else:
            lines.append([board[r][k] for r in range(3, -1, -1)])
    out = [[0] * 4 for _ in range(4)]
    points = 0
    for k, line in enumerate(lines):
        moved, p = slide_row_left(line)
        points += p
        for j, x in enumerate(moved):
            if move == 3:
                out[k][j] = x
            elif move == 1:
                out[k][3 - j] = x
            elif move == 0:
                out[j][k] = x
            else:
                out[3 - j][k] = x
    return out, points


def check_trace(boards, moves, score: int | None = None) -> str | None:
    """What in the trace breaks the rules of 2048, or None. boards: sequence of 4 x 4 exponent grids."""
    grids = [[[int(boards[t][r][c]) for c in range(4)] for r in range(4)] for t in range(len(boards))]
    moves = [int(m) for m in moves]
    if not grids or len(grids) != len(moves) or moves[-1] != -1 or any(m not in (0, 1, 2, 3) for m in moves[:-1]):
        return "the moves are not a finished game"
    first = [x for row in grids[0] for x in row if x]
    if len(first) != 2 or any(x not in (1, 2) for x in first):
        return "the game does not start with exactly two tiles, each a 2 or a 4"
    total = 0
    for t in range(len(grids) - 1):
        after, p = apply_move(grids[t], moves[t])
        if after == grids[t]:
            return f"move {t} does not change the board"
        total += p
        new = [(r, c) for r in range(4) for c in range(4) if grids[t + 1][r][c] != after[r][c]]
        if len(new) != 1:
            return f"board {t + 1} differs from board {t} after its move in {len(new)} cells, not one new tile"
        r, c = new[0]
        if after[r][c] != 0 or grids[t + 1][r][c] not in (1, 2):
            return f"board {t + 1}: the new tile is not a 2 or 4 in an empty cell"
    last = grids[-1]
    if any(apply_move(last, m)[0] != last for m in range(4)):
        return "the last board still has a move"
    if score is not None and total != int(score):
        return f"the moves score {total}, not {int(score)}"
    return None


if __name__ == "__main__":
    import numpy as np

    z = np.load(sys.argv[1])
    claimed = int(sys.argv[2]) if len(sys.argv) > 2 else int(z["score"])
    why = check_trace(z["boards"], z["dirs"], claimed)
    print("OK: every move follows the rules, score", claimed) if why is None else print("FAIL:", why)
    sys.exit(0 if why is None else 1)
