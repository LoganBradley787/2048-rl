import random

import pytest

from game2048.core import Direction, Game, slide_and_merge_row


# --- slide_and_merge_row: the single primitive every direction is built on ---

@pytest.mark.parametrize(
    "row, expected, reward",
    [
        ([0, 0, 0, 0], [0, 0, 0, 0], 0),
        ([2, 0, 0, 0], [2, 0, 0, 0], 0),
        ([0, 0, 0, 2], [2, 0, 0, 0], 0),
        ([0, 2, 0, 2], [4, 0, 0, 0], 4),
        ([2, 2, 2, 2], [4, 4, 0, 0], 8),
        ([2, 2, 2, 0], [4, 2, 0, 0], 4),      # leftmost pair merges, not the right
        ([4, 4, 8, 0], [8, 8, 0, 0], 8),      # a freshly merged 8 does not merge again
        ([2, 4, 2, 4], [2, 4, 2, 4], 0),
        ([8, 0, 8, 2], [16, 2, 0, 0], 16),
    ],
)
def test_slide_and_merge_row(row, expected, reward):
    assert slide_and_merge_row(row) == (expected, reward)


def test_slide_and_merge_row_does_not_mutate_input():
    row = [2, 2, 0, 0]
    slide_and_merge_row(row)
    assert row == [2, 2, 0, 0]


# --- Game construction ---

def test_new_game_has_two_tiles_and_zero_score():
    g = Game(seed=1)
    tiles = [v for r in g.board for v in r if v]
    assert len(tiles) == 2
    assert all(v in (2, 4) for v in tiles)
    assert g.score == 0


def test_same_seed_gives_same_game():
    a, b = Game(seed=42), Game(seed=42)
    assert a.board == b.board
    for _ in range(20):
        d = Direction(_ % 4)
        assert a.move(d) == b.move(d)
        assert a.board == b.board


# --- Moves in each direction ---

def board_from(rows):
    g = Game(seed=0)
    g.board = [list(r) for r in rows]
    g.score = 0
    return g


def _tiles_without_spawn(before, after):
    """Return `after` with the single spawned tile removed (set to 0)."""
    diff = [
        (i, j)
        for i in range(4)
        for j in range(4)
        if before[i][j] == 0 and after[i][j] != 0
    ]
    assert len(diff) == 1, "expected exactly one spawned tile"
    i, j = diff[0]
    out = [list(r) for r in after]
    out[i][j] = 0
    return out


def test_move_left():
    g = board_from([[2, 2, 0, 0], [0, 4, 0, 4], [8, 0, 0, 0], [0, 0, 0, 0]])
    res = g.move(Direction.LEFT)
    assert res.moved and res.reward == 12
    expected = [[4, 0, 0, 0], [8, 0, 0, 0], [8, 0, 0, 0], [0, 0, 0, 0]]
    assert _tiles_without_spawn(expected, g.board) == expected
    assert g.score == 12


def test_move_right():
    g = board_from([[2, 2, 0, 0], [0, 4, 0, 4], [8, 0, 0, 0], [0, 0, 0, 0]])
    res = g.move(Direction.RIGHT)
    assert res.moved and res.reward == 12
    expected = [[0, 0, 0, 4], [0, 0, 0, 8], [0, 0, 0, 8], [0, 0, 0, 0]]
    assert _tiles_without_spawn(expected, g.board) == expected


def test_move_up():
    g = board_from([[2, 0, 0, 0], [2, 4, 0, 0], [0, 0, 0, 0], [0, 4, 0, 8]])
    res = g.move(Direction.UP)
    assert res.moved and res.reward == 12
    expected = [[4, 8, 0, 8], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    assert _tiles_without_spawn(expected, g.board) == expected


def test_move_down():
    g = board_from([[2, 0, 0, 0], [2, 4, 0, 0], [0, 0, 0, 0], [0, 4, 0, 8]])
    res = g.move(Direction.DOWN)
    assert res.moved and res.reward == 12
    expected = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 8, 0, 8]]
    assert _tiles_without_spawn(expected, g.board) == expected


def test_move_accepts_plain_ints():
    g = board_from([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert g.move(3).moved  # 3 == LEFT


def test_invalid_move_changes_nothing_and_spawns_nothing():
    rows = [[2, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    g = board_from(rows)
    res = g.move(Direction.LEFT)
    assert res == (False, 0)
    assert g.board == rows
    assert g.score == 0


# --- Spawning ---

def test_spawn_fills_exactly_one_empty_cell_after_a_valid_move():
    g = board_from([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    g.move(Direction.LEFT)
    tiles = [v for r in g.board for v in r if v]
    assert sorted(tiles) in ([2, 4], [4, 4])


def test_spawn_is_2_about_ninety_percent_of_the_time():
    fours = 0
    n = 3000
    for seed in range(n):
        g = board_from([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        g._rng = random.Random(seed)
        g.move(Direction.LEFT)
        spawned = [v for r in g.board for v in r if v]
        spawned.remove(4)  # the merged tile
        fours += spawned[0] == 4
    assert 0.07 < fours / n < 0.13


# --- Legal moves / game over ---

def test_legal_moves_lists_only_directions_that_change_the_board():
    g = board_from([[2, 4, 8, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert g.legal_moves() == [Direction.DOWN]


def test_is_over_when_no_moves_remain():
    g = board_from([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    assert g.legal_moves() == []
    assert g.is_over()


def test_not_over_when_a_merge_is_possible_on_full_board():
    g = board_from([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 8, 8]])
    assert not g.is_over()
    assert set(g.legal_moves()) == {Direction.LEFT, Direction.RIGHT}


# --- Misc API ---

def test_max_tile():
    g = board_from([[2, 4, 0, 0], [0, 128, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert g.max_tile() == 128


def test_clone_is_independent_including_rng():
    g = Game(seed=7)
    c = g.clone()
    assert c.board == g.board and c.score == g.score
    g.move(Direction.LEFT)
    g.move(Direction.UP)
    assert c.board != g.board
    # clone's RNG was copied, so replaying the same moves gives the same result
    c.move(Direction.LEFT)
    c.move(Direction.UP)
    assert c.board == g.board


def test_to_dict_and_from_dict_round_trip():
    g = board_from([[2, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    g.score = 99
    d = g.to_dict()
    assert d["board"] == g.board
    assert d["score"] == 99
    assert d["over"] is False
    assert d["max_tile"] == 4
    assert d["legal_moves"] == [int(m) for m in g.legal_moves()]
    g2 = Game.from_dict(d)
    assert g2.board == g.board and g2.score == 99


def test_reset_starts_a_fresh_game():
    g = Game(seed=3)
    g.move(Direction.LEFT)
    g.score = 500
    g.reset()
    assert g.score == 0
    assert sum(1 for r in g.board for v in r if v) == 2


# --- spawn modes ---------------------------------------------------------------

def test_default_mode_is_random_and_reported_in_state():
    g = Game(seed=1)
    assert g.spawn_mode == "random"
    assert g.to_dict()["mode"] == "random"


def test_unknown_spawn_mode_is_rejected():
    with pytest.raises(ValueError):
        Game(seed=1, spawn_mode="friendly")


def test_kind_mode_never_ends_the_game_when_a_safe_spawn_exists():
    # Only (3, 3) is empty; a 4 there leaves no move, a 2 there merges with the 2 above it.
    rows = [[2, 4, 8, 16], [16, 8, 4, 32], [2, 4, 8, 2], [4, 2, 16, 0]]
    stuck_with_4 = [r[:] for r in rows]; stuck_with_4[3][3] = 4
    assert Game.from_dict({"board": stuck_with_4}).is_over()  # sanity: the 4 would end it
    for seed in range(40):
        g = Game(seed=seed, spawn_mode="kind")
        g.board = [r[:] for r in rows]
        g.score = 0
        g._spawn()
        assert g.board[3][3] == 2, seed
        assert not g.is_over()


def test_kind_mode_falls_back_to_random_when_every_spawn_ends_the_game():
    rows = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 8], [4, 2, 8, 0]]  # any tile at (3,3) is fatal
    g = Game(seed=3, spawn_mode="kind")
    g.board = [r[:] for r in rows]
    g._spawn()
    assert g.board[3][3] in (2, 4) and g.is_over()


def test_kind_mode_keeps_the_usual_2_to_4_ratio_when_both_are_safe():
    fours = 0
    for seed in range(400):
        g = Game(seed=seed, spawn_mode="kind")
        g.board = [[2, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        g._spawn()
        fours += sum(v == 4 for r in g.board for v in r) - 1
    assert 0.06 < fours / 400 < 0.15


def test_best_mode_picks_the_spawn_that_maximises_the_players_best_reply():
    # With a zero evaluator only the immediate merge counts: the best spawn is a 2 that can
    # merge with the lone 2, i.e. anywhere in row 0 or column 0.
    for seed in range(20):
        g = Game(seed=seed, spawn_mode="best", evaluator=lambda board: 0.0)
        g.board = [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        g._spawn()
        cells = [(i, j) for i in range(4) for j in range(4) if g.board[i][j] and (i, j) != (0, 0)]
        assert len(cells) == 1
        (i, j), = cells
        assert g.board[i][j] == 2 and (i == 0 or j == 0)


def test_best_mode_uses_the_evaluator_on_the_afterstate():
    # Evaluator rewards a big value in the top-left corner; after RIGHT/DOWN moves nothing is
    # there, so the chosen spawn + best reply must keep 2 at (0,0): a spawn in row 0 with LEFT.
    g = Game(seed=1, spawn_mode="best", evaluator=lambda board: 1000.0 * (board[0][0] == 4))
    g.board = [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    g._spawn()
    assert any(g.board[0][j] == 2 for j in range(1, 4))  # a 2 in row 0, so LEFT makes a 4 at (0,0)


def test_spawn_mode_survives_clone_and_dict_round_trip():
    g = Game(seed=1, spawn_mode="kind")
    assert g.clone().spawn_mode == "kind"
    assert Game.from_dict(g.to_dict()).spawn_mode == "kind"


def test_evil_mode_ends_the_game_whenever_a_spawn_can():
    rows = [[2, 4, 8, 16], [16, 8, 4, 32], [2, 4, 8, 2], [4, 2, 16, 0]]  # a 4 at (3,3) is fatal, a 2 is not
    for seed in range(10):
        g = Game(seed=seed, spawn_mode="evil", evaluator=lambda board: 0.0)
        g.board = [r[:] for r in rows]
        g._spawn()
        assert g.board[3][3] == 4 and g.is_over()


def test_evil_mode_avoids_giving_a_merge():
    for seed in range(20):
        g = Game(seed=seed, spawn_mode="evil", evaluator=lambda board: 0.0)
        g.board = [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        g._spawn()
        best_reward = max((g._apply(g.board, d)[1] for d in Direction), default=0)
        assert best_reward == 0


# --- choose mode: the player places every new tile -------------------------------

def test_choose_mode_waits_for_a_placement_after_a_move():
    g = Game(seed=1, spawn_mode="choose")
    assert g.awaiting_tile is False and sum(1 for r in g.board for v in r if v) == 2
    g.board = [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    res = g.move(Direction.LEFT)
    assert res == (True, 4)
    assert g.board == [[4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]  # nothing spawned
    assert g.awaiting_tile is True
    assert g.legal_moves() == [] and not g.is_over()
    with pytest.raises(RuntimeError, match="place"):
        g.move(Direction.UP)


def test_place_fills_an_empty_cell_and_resumes_play():
    g = Game(seed=1, spawn_mode="choose")
    g.board = [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    g.move(Direction.LEFT)
    g.place(0, 1, 4)
    assert g.board[0][1] == 4 and g.awaiting_tile is False
    assert Direction.UP not in g.legal_moves() and Direction.LEFT in g.legal_moves()


def test_best_placement_suggests_an_empty_cell_without_placing():
    g = Game(seed=1, spawn_mode="choose", evaluator=lambda board: 0.0)
    g.board = [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    g.awaiting_tile = True
    row, col, value = g.best_placement()
    assert g.board[row][col] == 0 and value == 2 and (row == 0 or col == 0)   # a merge is available next
    assert g.awaiting_tile is True and sum(v for r in g.board for v in r) == 2


def test_place_rejects_bad_placements():
    g = Game(seed=1, spawn_mode="choose")
    with pytest.raises(ValueError, match="not waiting"):
        g.place(3, 3, 2)
    g.board = [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    g.move(Direction.LEFT)
    with pytest.raises(ValueError, match="empty"):
        g.place(0, 0, 2)
    with pytest.raises(ValueError, match="2 or 4"):
        g.place(1, 1, 8)
    with pytest.raises(ValueError):
        g.place(4, 0, 2)
    assert g.awaiting_tile is True


def test_invalid_move_in_choose_mode_does_not_wait():
    g = Game(seed=1, spawn_mode="choose")
    g.board = [[2, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    assert g.move(Direction.LEFT) == (False, 0)
    assert g.awaiting_tile is False


def test_a_fatal_placement_ends_the_game():
    g = Game(seed=1, spawn_mode="choose")
    g.board = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 8], [4, 2, 8, 16]]
    g.board[3][3] = 0
    g.board[3][2] = 0
    # RIGHT slides row 3 (4, 2, _, _) to (_, _, 4, 2)... use a simple valid move instead
    g.board = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 8], [0, 4, 2, 8]]
    assert g.move(Direction.RIGHT) == (False, 0)  # already packed right
    assert g.move(Direction.LEFT) == (True, 0)   # row 3 slides: 4, 2, 8, _
    g.place(3, 3, 2)                              # neighbours 8 (left) and 8 (above): no merge anywhere
    assert g.is_over()
    assert g.to_dict()["over"] is True


def test_choose_mode_survives_dict_round_trip_and_clone():
    g = Game(seed=1, spawn_mode="choose")
    g.board = [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    g.move(Direction.LEFT)
    d = g.to_dict()
    assert d["mode"] == "choose" and d["awaiting_tile"] is True and d["legal_moves"] == []
    assert Game.from_dict(d).awaiting_tile is True
    c = g.clone()
    assert c.awaiting_tile is True
    c.place(2, 2, 2)
    assert g.awaiting_tile is True and g.board[2][2] == 0
