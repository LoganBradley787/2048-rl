import numpy as np
import pytest

from game2048 import ntuple as nt
from game2048 import vec_env as ve
from game2048.core import Direction, Game


def rand_exps(rng, density=0.6, max_exp=11):
    b = rng.integers(1, max_exp + 1, size=(4, 4)).astype(np.uint8)
    b[rng.random((4, 4)) > density] = 0
    return b


def to_values(b):
    return [[int(2 ** int(v)) if v else 0 for v in row] for row in b]


@pytest.fixture(scope="module")
def net():
    return nt.NTupleNet()


# --- bitboard engine ---------------------------------------------------------

def test_board_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        b = rand_exps(rng, density=1.0, max_exp=15)
        assert np.array_equal(nt.from_bits(nt.to_bits(b)), b)


def test_moves_match_game_rules_in_every_direction():
    rng = np.random.default_rng(1)
    for _ in range(300):
        b = rand_exps(rng)
        vals = to_values(b)
        for d in Direction:
            exp_board, exp_reward = Game._apply(vals, d)
            new_bits, reward = nt.move(nt.to_bits(b), int(d))
            assert to_values(nt.from_bits(new_bits)) == exp_board, (b, d)
            assert reward == exp_reward


def test_spawn_adds_one_two_or_four():
    rng = np.random.default_rng(2)
    fours = 0
    for seed in range(400):
        b = rand_exps(rng, density=0.5)
        after = nt.from_bits(nt.spawn(nt.to_bits(b), seed=seed))
        diff = (b == 0) & (after != 0)
        assert diff.sum() == 1 and after[diff][0] in (1, 2)
        assert np.array_equal(b[~diff], after[~diff])
        fours += after[diff][0] == 2
    assert 0.05 < fours / 400 < 0.16


# --- n-tuple network ---------------------------------------------------------

def test_fresh_network_values_are_zero(net):
    rng = np.random.default_rng(3)
    assert all(net.value(nt.to_bits(rand_exps(rng))) == 0.0 for _ in range(10))


def test_optimistic_network_values_every_board_at_v_init(tmp_path):
    rng = np.random.default_rng(4)
    boards = [nt.to_bits(rand_exps(rng, max_exp=15)) for _ in range(10)]
    for patterns in (nt.PATTERN_SETS["4x6"], nt.PATTERN_SETS["8x6"]):
        n = nt.NTupleNet(patterns=patterns, tc=False, v_init=320_000)
        assert all(n.value(b) == pytest.approx(320_000, rel=1e-5) for b in boards)
        n.save(tmp_path / "opt.bin")
        n.close()
        back = nt.NTupleNet.load(tmp_path / "opt.bin")
        assert back.value(boards[0]) == pytest.approx(320_000, rel=1e-5)
        back.close()


def test_optimistic_values_move_only_where_the_network_has_played():
    n = nt.NTupleNet(tc=False, v_init=10_000)
    n.train(200, threads=1, alpha=0.1, seed=1)
    start = nt.spawn(nt.spawn(0, 1), 2)
    after = next(a for a in (nt.move(start, d)[0] for d in range(4)) if a != start)
    unseen = nt.to_bits(np.full((4, 4), 15, dtype=np.uint8))
    assert abs(n.value(after) - 10_000) > 100          # every game passes through boards like this
    assert n.value(unseen) == pytest.approx(10_000, rel=1e-5)
    n.close()


def test_update_moves_value_by_alpha_times_delta():
    n = nt.NTupleNet(tc=False)
    # 16 distinct values so no two symmetric samples of a pattern share a table entry
    b = nt.to_bits(np.arange(16, dtype=np.uint8).reshape(4, 4))
    n.update(b, 100.0, alpha=0.5)
    assert n.value(b) == pytest.approx(50.0, rel=1e-5)
    n.close()


def test_value_is_invariant_under_board_symmetries():
    n = nt.NTupleNet(tc=False)
    rng = np.random.default_rng(5)
    for _ in range(5):
        n.update(nt.to_bits(rand_exps(rng)), rng.uniform(-50, 50), alpha=1.0)
    for _ in range(10):
        b = rand_exps(rng)
        base = n.value(nt.to_bits(b))
        for k in range(8):
            sym = ve.apply_symmetry(b[None], k)[0]
            assert n.value(nt.to_bits(sym)) == pytest.approx(base, abs=1e-3)
    n.close()


def test_training_is_deterministic_with_one_thread():
    a = nt.NTupleNet()
    b = nt.NTupleNet()
    sa = a.train(games=100, threads=1, seed=7)
    sb = b.train(games=100, threads=1, seed=7)
    assert np.array_equal(sa["scores"], sb["scores"]) and np.array_equal(sa["moves"], sb["moves"])
    assert sa["scores"].shape == (100,) and (sa["scores"] > 0).all()
    a.close(); b.close()


def test_training_improves_greedy_play():
    n = nt.NTupleNet()
    before = n.play(games=30, depth=0, threads=2, seed=11)["scores"].mean()
    n.train(games=3000, threads=4, seed=1)
    after = n.play(games=30, depth=0, threads=2, seed=11)["scores"].mean()
    assert after > 1.5 * before, (before, after)
    n.close()


# --- expectimax ---------------------------------------------------------------

def test_depth_zero_search_is_greedy_on_value():
    n = nt.NTupleNet()
    n.train(games=300, threads=2, seed=3)
    rng = np.random.default_rng(6)
    for _ in range(50):
        b = rand_exps(rng)
        bits = nt.to_bits(b)
        best, best_v = -1, None
        for d in range(4):
            nb, r = nt.move(bits, d)
            if nb != bits:
                v = r + n.value(nb)
                if best_v is None or v > best_v:
                    best, best_v = d, v
        assert n.best_move(bits, depth=0) == best
    n.close()


def _ref_expectimax(net, bits, depth):
    """Python reference: max over moves of reward + expectation over spawns."""
    def chance(after, depth):
        if depth == 0:
            return net.value(after)
        empties = [i for i in range(16) if nt.cell(after, i) == 0]
        total = 0.0
        for i in empties:
            total += 0.9 * maxnode(nt.with_cell(after, i, 1), depth - 1)
            total += 0.1 * maxnode(nt.with_cell(after, i, 2), depth - 1)
        return total / len(empties)

    def maxnode(b, depth):
        best = None
        for d in range(4):
            nb, r = nt.move(b, d)
            if nb != b:
                v = r + chance(nb, depth)
                best = v if best is None else max(best, v)
        return 0.0 if best is None else best

    return maxnode(bits, depth)


def test_search_value_is_exactly_the_reference_on_trained_and_stacked_networks():
    """The search batches its leaves and keeps them in a table of their own. It must still add
    the same numbers in the same order as the plain definition: equal, not approximately equal,
    for one network and for stacked stages, at any table size."""
    pats = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 4, 5]]
    a, b = nt.NTupleNet(patterns=pats, tc=False), nt.NTupleNet(patterns=pats, tc=False)
    a.train(400, threads=1, seed=1, alpha_plain=0.1)
    b.train(400, threads=1, seed=2, alpha_plain=0.1)
    both = nt.NTupleNet.stack([a, b], [64])
    rng = np.random.default_rng(21)
    boards = [nt.to_bits(rand_exps(rng, density=0.8, max_exp=8)) for _ in range(8)]
    for n in (a, both):
        for bits_ in (10, 20):
            n.tt_bits = bits_
            for bd in boards:
                assert n.search_value(bd, depth=1) == _ref_expectimax(n, bd, 1)
            for bd in boards[:2]:
                assert n.search_value(bd, depth=2) == _ref_expectimax(n, bd, 2)
    for n in (a, b, both):
        n.close()


def test_search_value_matches_python_reference(net):
    rng = np.random.default_rng(8)
    for _ in range(15):
        b = rand_exps(rng, density=0.8)
        bits = nt.to_bits(b)
        assert net.search_value(bits, depth=1) == pytest.approx(_ref_expectimax(net, bits, 1), rel=1e-6, abs=1e-6)
    for _ in range(3):
        b = rand_exps(rng, density=0.85)
        bits = nt.to_bits(b)
        assert net.search_value(bits, depth=2) == pytest.approx(_ref_expectimax(net, bits, 2), rel=1e-6, abs=1e-6)


def test_search_returns_minus_one_on_stuck_board(net):
    stuck = np.array([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]], dtype=np.uint8)
    assert net.best_move(nt.to_bits(stuck), depth=2) == -1


def test_play_reports_per_game_stats(net):
    stats = net.play(games=4, depth=1, threads=2, seed=9)
    assert stats["scores"].shape == (4,) and (stats["scores"] > 0).all()
    assert stats["max_tiles"].min() >= 8 and (stats["moves"] > 0).all()


# --- persistence and agent ------------------------------------------------------

def test_save_load_round_trip(tmp_path):
    a = nt.NTupleNet()
    a.train(games=200, threads=2, seed=2)
    path = tmp_path / "w.bin"
    a.save(path)
    b = nt.NTupleNet.load(path)
    rng = np.random.default_rng(10)
    for _ in range(20):
        bits = nt.to_bits(rand_exps(rng))
        assert b.value(bits) == a.value(bits)
    assert b.patterns == a.patterns and b.tc == a.tc
    a.close(); b.close()


def test_temporal_coherence_state_survives_save_and_load(tmp_path):
    """Values depend on the weights alone, so the round trip above would not notice the two
    error sums being lost or swapped. Training on from the file does: the per-entry rates
    come from those sums, so a copy with the wrong ones ends up with different weights."""
    a = nt.NTupleNet()
    a.train(games=60, threads=1, seed=2)
    path = tmp_path / "w.bin"
    a.save(path)
    b = nt.NTupleNet.load(path)
    a.train(games=60, threads=1, seed=5)
    b.train(games=60, threads=1, seed=5)
    rng = np.random.default_rng(11)
    for _ in range(40):
        bits = nt.to_bits(rand_exps(rng))
        assert b.value(bits) == a.value(bits)
    b.save(tmp_path / "w2.bin")
    a.save(tmp_path / "w1.bin")
    assert (tmp_path / "w1.bin").read_bytes() == (tmp_path / "w2.bin").read_bytes()
    a.close(); b.close()


def test_play_trace_is_the_same_game_as_play_with_its_history():
    n = nt.NTupleNet(patterns=[[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 4, 5]], tc=False)
    n.train(300, threads=1, seed=1, alpha_plain=0.1)
    for depth in (0, 1):
        for g in (0, 3):
            plain = n.play(1, depth=depth, threads=1, seed=7, first=g)
            tr = n.play_trace(g, depth=depth, seed=7)
            assert (tr["score"], tr["max_tile"], tr["moves"]) == (plain["scores"][0], plain["max_tiles"][0], plain["moves"][0])
            boards, dirs = tr["boards"], tr["dirs"]
            assert len(boards) == tr["moves"] + 1 and dirs[-1] == -1 and (dirs[:-1] >= 0).all()
            assert 1 << int(boards[-1].max()) == tr["max_tile"]
            score = 0
            for t in range(tr["moves"]):                   # each board follows from the one before by its move and one new tile
                after, r = nt.move(nt.to_bits(boards[t]), int(dirs[t]))
                score += r
                diff = boards[t + 1].astype(int) - nt.from_bits(after).astype(int)
                assert np.count_nonzero(diff) == 1 and diff.max() in (1, 2) and nt.from_bits(after)[diff != 0][0] == 0
            assert score == tr["score"]
            assert all(nt.move(nt.to_bits(boards[-1]), d)[0] == nt.to_bits(boards[-1]) for d in range(4))
    n.close()


def test_agent_plays_legal_moves(tmp_path):
    n = nt.NTupleNet()
    n.train(games=100, threads=2, seed=4)
    n.save(tmp_path / "w.bin")
    n.close()
    agent = nt.NTupleAgent(weights=tmp_path / "w.bin", depth=1)
    game = Game(seed=1)
    for _ in range(30):
        if game.is_over():
            break
        d = agent.act(game)
        assert d in game.legal_moves()
        game.move(d)


def test_agent_missing_weights_is_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="train"):
        nt.NTupleAgent(weights=tmp_path / "nope.bin")


# --- multi-stage tables ----------------------------------------------------------

def board_with(tiles):
    """Exponent grid with the given tile values placed row-major from the top left."""
    b = np.zeros((4, 4), dtype=np.uint8)
    for i, t in enumerate(tiles):
        b[i // 4, i % 4] = t.bit_length() - 1
    return b


def test_stage_is_number_of_boundaries_the_tiles_present_reach():
    bounds = [16384, 24576]
    assert nt.stage_of(nt.to_bits(board_with([8192, 4096, 2048])), bounds) == 0
    assert nt.stage_of(nt.to_bits(board_with([8192, 8192, 8192])), bounds) == 0     # 24576 in tiles, no 16384
    assert nt.stage_of(nt.to_bits(board_with([16384])), bounds) == 1
    assert nt.stage_of(nt.to_bits(board_with([16384, 8192])), bounds) == 2
    assert nt.stage_of(nt.to_bits(board_with([16384, 4096, 2048, 2048])), bounds) == 1  # no 8192
    assert nt.stage_of(nt.to_bits(board_with([16384, 16384])), bounds) == 1         # a second 16384 adds nothing
    assert nt.stage_of(nt.to_bits(board_with([32768])), bounds) == 2
    assert nt.stage_of(nt.to_bits(board_with([32768, 16384])), [16384, 49152]) == 2
    assert nt.stage_of(nt.to_bits(board_with([32768, 8192])), [16384, 49152]) == 1
    assert nt.stage_of(nt.to_bits(board_with([2])), []) == 0


def test_training_from_starts_begins_every_game_on_one_of_them():
    n = nt.NTupleNet(tc=False)
    start = nt.to_bits(board_with([4096, 2048, 1024, 512, 2, 4]))
    stats = n.train(games=40, threads=2, seed=3, alpha_plain=0.1, starts=[start])
    assert (stats["max_tiles"] >= 4096).all()                    # the 4096 was there from the first move
    assert stats["moves"].mean() < 600                           # and no game had to build it
    fresh = nt.NTupleNet(tc=False).train(games=40, threads=2, seed=3, alpha_plain=0.1)
    assert (fresh["max_tiles"] < 4096).all()


def test_training_without_starts_is_unchanged_by_the_starts_argument():
    a, b = nt.NTupleNet(tc=False), nt.NTupleNet(tc=False)
    sa = a.train(games=30, threads=1, seed=5, alpha_plain=0.1)
    sb = b.train(games=30, threads=1, seed=5, alpha_plain=0.1, starts=[])
    assert (sa["scores"] == sb["scores"]).all()


def test_collect_starts_keeps_the_first_board_with_the_tile():
    n = nt.NTupleNet(tc=False)
    n.train(games=300, threads=2, seed=1, alpha_plain=0.1)
    grids, scores = n.collect_starts(games=20, tile=128, threads=2, seed=9)
    assert grids.shape[1:] == (4, 4) and len(grids) == len(scores) > 0
    for g in grids:
        assert g.max() == 7                                      # 128 itself: the first time one appears
        assert (g == 7).sum() == 1
    assert (scores > 0).all()
    again, _ = n.collect_starts(games=20, tile=128, threads=1, seed=9)
    assert (again == grids).all()                                # the seed fixes the games, not the thread count
    none, _ = n.collect_starts(games=5, tile=131072, threads=1, seed=9)
    assert len(none) == 0


def test_stack_plays_each_board_with_its_own_stage_network():
    pats = [[0, 1, 2, 3], [4, 5, 6, 7]]
    a, b = nt.NTupleNet(patterns=pats, tc=False, v_init=100.0), nt.NTupleNet(patterns=pats, tc=False, v_init=900.0)
    s = nt.NTupleNet.stack([a, b], [16384])
    assert s.boundaries == [16384] and s.tc is False
    lo, hi = nt.to_bits(board_with([8192, 4])), nt.to_bits(board_with([16384, 4]))
    assert s.value(lo) == pytest.approx(100.0) and s.value(hi) == pytest.approx(900.0)
    b.update(hi, 50.0, alpha_plain=1.0)                          # the stack holds copies
    assert s.value(hi) == pytest.approx(900.0)
    with pytest.raises(ValueError):
        nt.NTupleNet.stack([a, nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False)], [16384])
    with pytest.raises(ValueError):
        nt.NTupleNet.stack([a, b], [])


def test_stacked_network_survives_a_save_and_load(tmp_path):
    pats = [[0, 1, 2, 3]]
    a, b = nt.NTupleNet(patterns=pats, tc=False, v_init=10.0), nt.NTupleNet(patterns=pats, tc=False, v_init=70.0)
    nt.NTupleNet.stack([a, b], [16384]).save(tmp_path / "s.bin", weights_only=True)
    m = nt.NTupleNet.load(tmp_path / "s.bin")
    assert m.boundaries == [16384]
    assert m.value(nt.to_bits(board_with([16384]))) == pytest.approx(70.0)
    assert m.value(nt.to_bits(board_with([8192]))) == pytest.approx(10.0)


def test_pack_starts_packs_grids_like_bitboards():
    rng = np.random.default_rng(0)
    grids = rng.integers(0, 18, size=(50, 4, 4), dtype=np.uint8)
    fast, n = nt.pack_starts(grids)
    slow, m = nt.pack_starts([nt.to_bits(g) for g in grids])
    assert n == m == 50 and list(fast) == list(slow)
    assert nt.pack_starts(np.zeros((0, 4, 4), dtype=np.uint8))[1] == 0


def test_percent_rounds_down():
    assert nt.percent(0.0055) == "0%" and nt.percent(0.0055, 1) == "0.5%"
    assert nt.percent(0.01) == "1%" and nt.percent(0.999) == "99%" and nt.percent(1.0) == "100%"
    assert nt.percent(0.29) == "29%" and nt.percent(0.0141, 1) == "1.4%"   # 0.29 * 100 is 28.999... in floating point


def test_summarize_reports_the_65536_rate():
    stats = {"scores": np.array([10, 20]), "max_tiles": np.array([65536, 32768]), "moves": np.array([5, 6])}
    assert nt.summarize(stats)["rate_65536"] == 0.5


def test_untouched_stage_entries_read_through_from_the_previous_stage():
    n = nt.NTupleNet(boundaries=[60000])
    # distinct cell values so no two symmetric samples of a pattern share an entry
    z = np.arange(16, dtype=np.uint8).reshape(4, 4)
    z[3, 3] = 0                   # mass 32766 -> stage 0
    zb = nt.to_bits(z)
    n.update(zb, 100.0, alpha=1.0)
    assert n.value_in_stage(zb, 0) == pytest.approx(100.0, rel=1e-5)
    assert n.value_in_stage(zb, 1) == pytest.approx(100.0, rel=1e-5)  # promoted from stage 0
    n.update_in_stage(zb, 50.0, alpha=1.0, stage=1)
    assert n.value_in_stage(zb, 1) == pytest.approx(150.0, rel=1e-5)
    assert n.value_in_stage(zb, 0) == pytest.approx(100.0, rel=1e-5)  # stage 0 untouched by that
    assert n.value(zb) == n.value_in_stage(zb, 0)                     # natural stage 0
    a = nt.to_bits(np.arange(16, dtype=np.uint8).reshape(4, 4))       # mass 65534 -> stage 1
    assert nt.stage_of(a, [60000]) == 1
    assert n.value(a) == n.value_in_stage(a, 1)
    n.close()


def test_stages_survive_save_and_load(tmp_path):
    n = nt.NTupleNet(boundaries=[16384, 24576])
    n.train(games=50, threads=2, seed=1)
    n.save(tmp_path / "s.bin")
    m = nt.NTupleNet.load(tmp_path / "s.bin")
    assert m.boundaries == [16384, 24576]
    rng = np.random.default_rng(12)
    for _ in range(10):
        bits = nt.to_bits(rand_exps(rng))
        assert m.value(bits) == n.value(bits)
    n.close(); m.close()


def test_default_network_has_no_stages(net):
    assert net.boundaries == []


# --- weights-only deployment files ------------------------------------------------

def test_weights_only_save_is_smaller_and_plays_identically(tmp_path):
    n = nt.NTupleNet(boundaries=[60000])
    n.train(games=100, threads=2, seed=3)
    hi = nt.to_bits(np.arange(16, dtype=np.uint8).reshape(4, 4))  # mass 65534 -> stage 1
    n.update(hi, 25.0)                                              # touches some stage-1 entries
    n.save(tmp_path / "full.bin")
    n.save(tmp_path / "wo.bin", weights_only=True)
    assert (tmp_path / "wo.bin").stat().st_size < (tmp_path / "full.bin").stat().st_size / 2
    m = nt.NTupleNet.load(tmp_path / "wo.bin")
    assert m.boundaries == [60000] and m.tc is False
    rng = np.random.default_rng(13)
    for _ in range(20):
        bits = nt.to_bits(rand_exps(rng))
        assert m.value(bits) == pytest.approx(n.value(bits), rel=1e-6, abs=1e-4)
    assert m.value(hi) == pytest.approx(n.value(hi), rel=1e-6, abs=1e-4)   # promotion baked in
    assert m.best_move(hi, depth=1) == n.best_move(hi, depth=1)
    n.close(); m.close()


def test_default_weights_resolve_to_first_existing_candidate(tmp_path, monkeypatch):
    monkeypatch.delenv("GAME2048_NTUPLE", raising=False)
    monkeypatch.setattr(nt, "DEFAULT_CANDIDATES", [tmp_path / "weights.bin", tmp_path / "latest.bin"])
    with pytest.raises(FileNotFoundError):
        nt.resolve_weights(None)
    (tmp_path / "latest.bin").write_bytes(b"x")
    assert nt.resolve_weights(None) == tmp_path / "latest.bin"
    (tmp_path / "weights.bin").write_bytes(b"x")
    assert nt.resolve_weights(None) == tmp_path / "weights.bin"
    assert nt.resolve_weights(tmp_path / "latest.bin") == tmp_path / "latest.bin"


def test_lock_memory_pins_tables_and_training_still_works():
    n = nt.NTupleNet()
    locked = n.lock_memory()      # False where the OS refuses (a container's memlock limit), never an error
    assert locked in (True, False)
    stats = n.train(games=50, threads=2, seed=1)
    assert (stats["scores"] > 0).all()
    n.close()


def test_single_stage_weights_can_seed_a_multistage_network(tmp_path):
    a = nt.NTupleNet()
    a.train(games=200, threads=2, seed=6)
    a.save(tmp_path / "single.bin")
    m = nt.NTupleNet.load(tmp_path / "single.bin", boundaries=[16384, 24576])
    assert m.boundaries == [16384, 24576] and m.tc
    assert m.tc_stages == 3   # a TC file re-staged keeps temporal coherence on every stage
    rng = np.random.default_rng(14)
    for _ in range(20):
        bits = nt.to_bits(rand_exps(rng))
        assert m.value(bits) == a.value(bits)
        assert m.value_in_stage(bits, 2) == a.value(bits)   # untouched stages read through
    hi = nt.to_bits(np.arange(16, dtype=np.uint8).reshape(4, 4))  # mass 65534 -> stage 2
    assert m.value(hi) == a.value(hi)
    stats = m.train(games=50, threads=2, seed=7)                  # learning continues
    assert (stats["scores"] > 0).all()
    a.close(); m.close()


def test_loading_with_matching_boundaries_is_a_plain_load(tmp_path):
    a = nt.NTupleNet(boundaries=[60000])
    a.train(games=50, threads=2, seed=8)
    a.save(tmp_path / "s.bin")
    m = nt.NTupleNet.load(tmp_path / "s.bin", boundaries=[60000])
    assert m.boundaries == [60000]
    with pytest.raises(ValueError):
        nt.NTupleNet.load(tmp_path / "s.bin", boundaries=[16384])  # cannot re-stage a staged file
    a.close(); m.close()


# --- memory-light multi-stage: plain TD later stages with touched bits ------------

def test_plain_multistage_net_keeps_only_a_touched_bit_per_entry(tmp_path):
    n = nt.NTupleNet(tc=False, boundaries=[60000])
    n.train(games=30, threads=2, seed=1)
    n.save(tmp_path / "plain.bin")
    weights_bytes = 2 * 4 * (18 ** 6) * 4          # 2 stages x 4 patterns x 18^6 float32
    assert (tmp_path / "plain.bin").stat().st_size < weights_bytes + 32 * 2 ** 20
    m = nt.NTupleNet.load(tmp_path / "plain.bin")
    assert m.boundaries == [60000] and m.tc is False
    rng = np.random.default_rng(15)
    for _ in range(10):
        bits = nt.to_bits(rand_exps(rng))
        assert m.value(bits) == n.value(bits)
    n.close(); m.close()


def test_plain_multistage_promotion_reads_through_and_then_diverges():
    n = nt.NTupleNet(tc=False, boundaries=[60000])
    z = np.arange(16, dtype=np.uint8).reshape(4, 4)
    z[3, 3] = 0
    zb = nt.to_bits(z)
    n.update(zb, 100.0, alpha=1.0)
    assert n.value_in_stage(zb, 1) == pytest.approx(100.0, rel=1e-5)
    n.update_in_stage(zb, 50.0, alpha=1.0, stage=1)
    assert n.value_in_stage(zb, 1) == pytest.approx(150.0, rel=1e-5)
    assert n.value_in_stage(zb, 0) == pytest.approx(100.0, rel=1e-5)
    n.close()


def test_tc_file_can_be_loaded_as_plain_multistage(tmp_path):
    a = nt.NTupleNet()                      # TC, single stage
    a.train(games=100, threads=2, seed=2)
    a.save(tmp_path / "tc.bin")
    m = nt.NTupleNet.load(tmp_path / "tc.bin", boundaries=[16384, 24576], keep_tc=False)
    assert m.tc is False and m.boundaries == [16384, 24576]
    rng = np.random.default_rng(16)
    for _ in range(10):
        bits = nt.to_bits(rand_exps(rng))
        assert m.value(bits) == a.value(bits)
    m.save(tmp_path / "m.bin")
    assert (tmp_path / "m.bin").stat().st_size < 3 * 4 * (18 ** 6) * 4 + 48 * 2 ** 20
    stats = m.train(games=20, threads=2, seed=3)
    assert (stats["scores"] > 0).all()
    a.close(); m.close()


def test_short_patterns_keep_their_touched_bits_through_save_and_load(tmp_path):
    # 18 and 18**2 table entries are not multiples of 8: the last touched bits need a byte of their own.
    n = nt.NTupleNet(patterns=[[0], [1, 2], [0, 1, 2, 3]], tc=False, boundaries=[60000])
    top = np.full((4, 4), 17, dtype=np.uint8)            # every index is the table's last entry
    top[3] = [16, 15, 14, 13]                            # mass above the boundary: stage 1
    hb = nt.to_bits(top)
    assert nt.stage_of(hb, [60000]) == 1
    n.update(hb, 80.0, alpha=1.0)
    value = n.value(hb)                                  # symmetric samples share entries, so not simply 80
    assert value > 0
    n.save(tmp_path / "short.bin")
    n.save(tmp_path / "short_wo.bin", weights_only=True)  # mixed table sizes share one scratch buffer
    for name in ("short.bin", "short_wo.bin"):
        m = nt.NTupleNet.load(tmp_path / name)
        assert m.patterns == [[0], [1, 2], [0, 1, 2, 3]]
        assert m.value(hb) == pytest.approx(value, rel=1e-6)
        assert m.value_in_stage(hb, 0) == pytest.approx(0.0, abs=1e-6)   # stage 0 was never updated
        m.close()
    n.close()


# --- hybrid: temporal coherence on the first stages, plain TD on the rest ------------

def test_tc_stages_limits_learning_state_to_leading_stages(tmp_path):
    n = nt.NTupleNet(boundaries=[60000, 70000], tc_stages=1)
    assert n.tc_stages == 1 and n.tc and n.boundaries == [60000, 70000]
    n.train(games=30, threads=2, seed=1)
    n.save(tmp_path / "h.bin")
    one_table = 4 * (18 ** 6) * 4                       # 4 patterns x 18^6 x float32 = 256 MB
    size = (tmp_path / "h.bin").stat().st_size
    assert 3 * one_table + 2 * one_table < size < 3 * one_table + 2 * one_table + 64 * 2 ** 20  # w x3, e/a for stage 0 only
    m = nt.NTupleNet.load(tmp_path / "h.bin")
    assert m.tc_stages == 1 and m.boundaries == [60000, 70000]
    rng = np.random.default_rng(17)
    for _ in range(10):
        bits = nt.to_bits(rand_exps(rng))
        assert m.value(bits) == n.value(bits)
    n.close(); m.close()


def test_plain_stages_use_their_own_step_size():
    n = nt.NTupleNet(boundaries=[60000], tc_stages=1)
    z = np.arange(16, dtype=np.uint8).reshape(4, 4)
    z[3, 3] = 0
    zb = nt.to_bits(z)
    n.update(zb, 100.0, alpha=1.0, alpha_plain=0.5)          # stage 0 is TC: first touch applies the full delta
    assert n.value_in_stage(zb, 0) == pytest.approx(100.0, rel=1e-5)
    n.update_in_stage(zb, 100.0, alpha=1.0, alpha_plain=0.5, stage=1)   # promoted 100, then plain step 0.5 x 100
    assert n.value_in_stage(zb, 1) == pytest.approx(150.0, rel=1e-5)
    n.close()


def test_tc_file_loads_as_hybrid_multistage(tmp_path):
    a = nt.NTupleNet()
    a.train(games=100, threads=2, seed=2)
    a.save(tmp_path / "tc.bin")
    m = nt.NTupleNet.load(tmp_path / "tc.bin", boundaries=[16384, 24576, 32768], tc_stages=1)
    assert m.tc_stages == 1 and m.boundaries == [16384, 24576, 32768]
    rng = np.random.default_rng(18)
    for _ in range(10):
        bits = nt.to_bits(rand_exps(rng))
        assert m.value(bits) == a.value(bits)
    stats = m.train(games=20, threads=2, seed=3, alpha=1.0, alpha_plain=0.02)
    assert (stats["scores"] > 0).all()
    a.close(); m.close()


def test_promoted_tc_entries_inherit_the_parent_rate_state():
    n = nt.NTupleNet(boundaries=[60000])          # temporal coherence on both stages
    z = np.arange(16, dtype=np.uint8).reshape(4, 4)
    z[3, 3] = 0
    zb = nt.to_bits(z)
    n.update(zb, 100.0)                            # E=100, A=100 per entry -> value 100
    n.update(zb, -100.0)                           # E=0,   A=200 -> value back to 0, rate |E|/A = 0
    assert n.value_in_stage(zb, 0) == pytest.approx(0.0, abs=1e-4)
    n.update_in_stage(zb, 50.0, stage=1)           # promoted entries inherit E/A, so this step is damped to nothing
    assert n.value_in_stage(zb, 1) == pytest.approx(0.0, abs=1e-4)
    n.update_in_stage(zb, 50.0, stage=1)           # E=50, A=250 -> rate 0.2: moves by 0.2 * 50
    assert n.value_in_stage(zb, 1) == pytest.approx(10.0, rel=1e-4)
    assert n.value_in_stage(zb, 0) == pytest.approx(0.0, abs=1e-4)
    n.close()


# --- pattern sets ------------------------------------------------------------------

def test_pattern_sets_are_distinct_under_symmetry():
    for name, pats in nt.PATTERN_SETS.items():
        orbits = set()
        for p in pats:
            assert len(set(p)) == len(p) and (len(p) == 6 or (name.endswith("-redundant") and 3 <= len(p) < 6))
            orbit = frozenset(frozenset(nt.sym_cell(c, k) for c in p) for k in range(8))
            assert orbit not in orbits, f"{name}: {p} repeats another pattern's symmetry orbit"
            orbits.add(orbit)
    assert nt.PATTERN_SETS["4x6"] == nt.DEFAULT_PATTERNS and len(nt.PATTERN_SETS["8x6"]) == 8
    extra = nt.PATTERN_SETS["8x6-matsuzaki-redundant"]
    assert extra[:8] == nt.PATTERN_SETS["8x6-matsuzaki"] and sorted(len(p) for p in extra[8:]) == [3, 3, 4, 4, 4, 4, 4]


def test_bigger_pattern_set_trains_and_round_trips(tmp_path):
    n = nt.NTupleNet(patterns=nt.PATTERN_SETS["8x6"])
    stats = n.train(games=50, threads=2, seed=1)
    assert (stats["scores"] > 0).all()
    n.save(tmp_path / "p8.bin", weights_only=True)
    m = nt.NTupleNet.load(tmp_path / "p8.bin")
    assert m.patterns == nt.PATTERN_SETS["8x6"]
    n.close(); m.close()


# --- choose mode: max-max search (the player places every tile) -------------------

def _ref_choose(net, bits, depth):
    """Python reference. MV(s, d) = max over moves of reward + AV(after, d);
    AV(a, 0) = V(a); AV(a, d) = max over placements of MV(placed, d - 1). Dead board = 0."""
    def av(a, d):
        if d == 0:
            return net.value(a)
        best = None
        for i in range(16):
            if nt.cell(a, i):
                continue
            for v in (1, 2):
                val = mv(nt.with_cell(a, i, v), d - 1)
                best = val if best is None else max(best, val)
        return 0.0 if best is None else best

    def mv(s, d):
        best = None
        for m in range(4):
            a, r = nt.move(s, m)
            if a != s:
                val = r + av(a, d)
                best = val if best is None else max(best, val)
        return 0.0 if best is None else best

    return mv(bits, depth)


def test_choose_value_matches_python_reference():
    n = nt.NTupleNet()
    n.train(games=200, threads=2, seed=4)
    rng = np.random.default_rng(21)
    for _ in range(12):
        b = rand_exps(rng, density=0.85)          # few empty cells keep the reference tree small
        bits = nt.to_bits(b)
        assert n.choose_value(bits, depth=1, topk=32) == pytest.approx(_ref_choose(n, bits, 1), rel=1e-6, abs=1e-6)
    for _ in range(3):
        b = rand_exps(rng, density=0.9)
        bits = nt.to_bits(b)
        assert n.choose_value(bits, depth=2, topk=32) == pytest.approx(_ref_choose(n, bits, 2), rel=1e-6, abs=1e-6)
    n.close()


def test_best_choose_returns_a_move_and_placement_that_achieve_the_value():
    n = nt.NTupleNet()
    n.train(games=200, threads=2, seed=5)
    rng = np.random.default_rng(22)
    for _ in range(10):
        bits = nt.to_bits(rand_exps(rng, density=0.8))
        move, cell, value = n.best_choose(bits, depth=1, topk=32)
        if move < 0:
            assert nt.afterstates_valid(bits) == []
            continue
        after, reward = nt.move(bits, move)
        assert after != bits and nt.cell(after, cell) == 0 and value in (1, 2)
        placed = nt.with_cell(after, cell, value)
        assert reward + _ref_choose(n, placed, 0) == pytest.approx(n.choose_value(bits, depth=1, topk=32), rel=1e-6, abs=1e-6)
    stuck = nt.to_bits(np.array([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]], dtype=np.uint8))
    assert n.best_choose(stuck, depth=2)[0] == -1
    n.close()


def test_play_choose_scores_far_above_random_spawns():
    # One training thread: several threads race on the tables, so the same seed
    # gives different tables on every run, and with them this ratio ranged from
    # under 1.5 to 2.8. Single-threaded it is the same 2.1 every time.
    n = nt.NTupleNet()
    n.train(games=2000, threads=1, seed=7)
    normal = n.play(games=16, depth=1, threads=4, seed=3)["scores"].mean()
    stats = n.play_choose(games=8, depth=1, topk=4, threads=4, seed=3, max_moves=3000)
    assert stats["scores"].shape == (8,) and (stats["moves"] > 0).all()
    assert stats["scores"].mean() > 1.5 * normal, (normal, stats["scores"].mean())
    n.close()


def test_ntuple_agent_chooses_move_and_placement(tmp_path):
    n = nt.NTupleNet()
    n.train(games=100, threads=2, seed=7)
    n.save(tmp_path / "w.bin")
    n.close()
    agent = nt.NTupleAgent(weights=tmp_path / "w.bin", depth=1)
    game = Game(seed=2, spawn_mode="choose")
    for _ in range(20):
        direction, (row, col, value) = agent.choose(game)
        assert direction in game.legal_moves() and value in (2, 4)
        game.move(direction)
        game.place(row, col, value)          # raises if the cell was not empty
    assert game.score > 0


# --- learning the choose game -----------------------------------------------------

def test_train_choose_learns_to_place_tiles_well():
    # Games run to their natural end: cutting many games at a move cap leaves the
    # late boards ungrounded, values inflate and the policy collapses.
    # The training games are the measurement: each is played by the policy as it
    # stood, so their scores rise as it learns. Evaluating the trained tables on
    # the choose game instead would test one line of play (a deterministic policy
    # has only one), and where that line ends is chaotic: the same run scored
    # 65,668 on one build and 27,956 on another that differed only in whether a
    # multiply-add was fused. One thread keeps the run deterministic.
    n = nt.NTupleNet()
    stats = n.train_choose(games=600, threads=1, seed=1, max_moves=100_000)
    scores = stats["scores"]
    assert scores.shape == (600,) and (stats["moves"] > 0).all()
    early, late = scores[:100].mean(), scores[-300:].mean()
    assert late > 1.3 * early, (early, late)
    n.close()


def test_train_choose_is_deterministic_with_one_thread():
    a, b = nt.NTupleNet(), nt.NTupleNet()
    sa = a.train_choose(games=20, threads=1, seed=3, max_moves=500)
    sb = b.train_choose(games=20, threads=1, seed=3, max_moves=500)
    assert np.array_equal(sa["scores"], sb["scores"])
    a.close(); b.close()


# --- beam search for the choose game ------------------------------------------------

def test_beam_value_with_unlimited_width_matches_the_full_tree():
    n = nt.NTupleNet()
    n.train(games=200, threads=2, seed=4)
    rng = np.random.default_rng(31)
    for _ in range(12):
        bits = nt.to_bits(rand_exps(rng, density=0.85))
        assert n.beam_value(bits, width=4096, depth=1) == pytest.approx(n.choose_value(bits, depth=1, topk=32), rel=1e-6, abs=1e-6)
    for _ in range(3):
        bits = nt.to_bits(rand_exps(rng, density=0.9))
        assert n.beam_value(bits, width=4096, depth=2) == pytest.approx(n.choose_value(bits, depth=2, topk=32), rel=1e-6, abs=1e-6)
    n.close()


def test_beam_choose_returns_the_first_step_of_the_best_line():
    n = nt.NTupleNet()
    n.train(games=200, threads=2, seed=5)
    rng = np.random.default_rng(32)
    for _ in range(10):
        bits = nt.to_bits(rand_exps(rng, density=0.8))
        move, cell, value = n.beam_choose(bits, width=4096, depth=1)
        if move < 0:
            assert nt.afterstates_valid(bits) == []
            continue
        after, reward = nt.move(bits, move)
        assert after != bits and nt.cell(after, cell) == 0 and value in (1, 2)
        placed = nt.with_cell(after, cell, value)
        assert reward + _ref_choose(n, placed, 0) == pytest.approx(n.beam_value(bits, width=4096, depth=1), rel=1e-6, abs=1e-6)
    stuck = nt.to_bits(np.array([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]], dtype=np.uint8))
    assert n.beam_choose(stuck, width=8, depth=4)[0] == -1
    n.close()


def test_play_beam_reports_stats_and_is_deterministic():
    n = nt.NTupleNet()
    n.train_choose(games=200, threads=2, seed=6, max_moves=100_000)
    a = n.play_beam(games=4, width=8, depth=6, threads=2, seed=3, max_moves=1500)
    b = n.play_beam(games=4, width=8, depth=6, threads=2, seed=3, max_moves=1500)
    assert a["scores"].shape == (4,) and (a["moves"] > 0).all() and (a["scores"] > 0).all()
    assert np.array_equal(a["scores"], b["scores"]) and np.array_equal(a["moves"], b["moves"])
    n.close()


def test_ntuple_agent_can_use_beam_search(tmp_path):
    n = nt.NTupleNet()
    n.train(games=100, threads=2, seed=7)
    n.save(tmp_path / "w.bin")
    n.close()
    agent = nt.NTupleAgent(weights=tmp_path / "w.bin", beam_width=8, beam_depth=4)
    game = Game(seed=2, spawn_mode="choose")
    for _ in range(20):
        direction, (row, col, value) = agent.choose(game)
        assert direction in game.legal_moves() and value in (2, 4)
        game.move(direction)
        game.place(row, col, value)
    assert game.score > 0


def test_beam_plan_is_a_playable_line_worth_the_beam_value():
    n = nt.NTupleNet()
    n.train(games=200, threads=2, seed=8)
    rng = np.random.default_rng(33)
    for _ in range(6):
        bits = nt.to_bits(rand_exps(rng, density=0.6))
        plan = n.beam_plan(bits, width=16, depth=5)
        assert 1 <= len(plan) <= 5
        b, total = bits, 0
        for move, cell, value in plan:
            after, reward = nt.move(b, move)
            assert after != b and nt.cell(after, cell) == 0 and value in (1, 2)
            b = nt.with_cell(after, cell, value)
            total += reward
        assert total + _ref_choose(n, b, 0) == pytest.approx(n.beam_value(bits, width=16, depth=5), rel=1e-6, abs=1e-6)
    n.close()


def test_play_beam_stride_commits_several_steps_of_each_plan():
    n = nt.NTupleNet()
    n.train_choose(games=200, threads=2, seed=6, max_moves=100_000)
    one = n.play_beam(games=4, width=8, depth=6, threads=2, seed=3, max_moves=1500, stride=1)
    base = n.play_beam(games=4, width=8, depth=6, threads=2, seed=3, max_moves=1500)
    assert np.array_equal(one["scores"], base["scores"])
    four = n.play_beam(games=4, width=8, depth=6, threads=2, seed=3, max_moves=1500, stride=4)
    assert four["scores"].shape == (4,) and (four["scores"] > 0).all() and (four["moves"] > 0).all()
    n.close()


def test_ntuple_agent_follows_its_beam_plan_until_the_board_diverges(tmp_path):
    n = nt.NTupleNet()
    n.train(games=100, threads=2, seed=7)
    n.save(tmp_path / "w.bin")
    n.close()
    agent = nt.NTupleAgent(weights=tmp_path / "w.bin", beam_width=8, beam_depth=6, beam_stride=3)
    game = Game(seed=2, spawn_mode="choose")
    searches = 0
    real = agent.net.beam_plan

    def counting(*a, **k):
        nonlocal searches
        searches += 1
        return real(*a, **k)

    agent.net.beam_plan = counting
    for _ in range(9):
        direction, (row, col, value) = agent.choose(game)
        assert direction in game.legal_moves() and value in (2, 4)
        game.move(direction)
        game.place(row, col, value)
    assert searches == 3                      # one search per 3 steps
    game2 = Game(seed=5, spawn_mode="choose")  # a board the plan was not made for: search again
    direction, (row, col, value) = agent.choose(game2)
    assert direction in game2.legal_moves()
    assert searches == 4


def test_tiles_to_bits_never_mistakes_small_tiles_for_exponents():
    board = [[2, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 8]]
    assert nt.tiles_to_bits(board) == nt.with_cell(nt.with_cell(nt.with_cell(0, 0, 1), 1, 2), 15, 3)
    assert nt.tiles_to_bits([[0] * 4] * 4) == 0


def test_to_bits_takes_exponents_and_refuses_tile_values():
    assert nt.to_bits([[17, 1, 0, 0]] + [[0] * 4] * 3) == nt.with_cell(nt.with_cell(0, 0, 17), 1, 1)
    with pytest.raises(ValueError, match="tiles_to_bits"):
        nt.to_bits([[2048, 2, 0, 0]] + [[0] * 4] * 3)


def test_patterns_must_name_board_cells():
    with pytest.raises(ValueError, match="0..15"):
        nt.NTupleNet(patterns=[[0, 1, 16]], tc=False)


def test_beam_plan_with_a_zero_depth_plans_one_step():
    n = nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False)
    bits = nt.to_bits(rand_exps(np.random.default_rng(42), density=0.5))
    assert len(n.beam_plan(bits, width=4, depth=0, snake=1.0)) == 1
    n.close()


def test_agent_sees_the_real_board_early_in_a_game(tmp_path):
    n = nt.NTupleNet()
    n.train(games=100, threads=2, seed=7)
    n.save(tmp_path / "w.bin")
    n.close()
    agent = nt.NTupleAgent(weights=tmp_path / "w.bin", depth=1)
    game = Game(seed=2, spawn_mode="choose")
    game.board = [[2, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 8]]   # no tile above 15
    seen = []
    real_choose, real_move = agent.net.best_choose, agent.net.best_move
    agent.net.best_choose = lambda bits, *a: seen.append(bits) or real_choose(bits, *a)
    agent.net.best_move = lambda bits, *a: seen.append(bits) or real_move(bits, *a)
    agent.choose(game)
    agent.act(game)
    assert seen == [nt.tiles_to_bits(game.board)] * 2


def test_beam_spread_caps_children_per_parent_and_keeps_lines_consistent():
    n = nt.NTupleNet()
    n.train(games=200, threads=2, seed=8)
    rng = np.random.default_rng(34)
    for _ in range(6):
        bits = nt.to_bits(rand_exps(rng, density=0.6))
        assert n.beam_value(bits, width=16, depth=4, spread=0) == n.beam_value(bits, width=16, depth=4)
        assert n.beam_value(bits, width=1, depth=4, spread=1) == n.beam_value(bits, width=1, depth=4)
        plan = n.beam_plan(bits, width=16, depth=4, spread=1)
        assert 1 <= len(plan) <= 4
        b, total = bits, 0
        for move, cell, value in plan:
            after, reward = nt.move(b, move)
            assert after != b and nt.cell(after, cell) == 0
            b = nt.with_cell(after, cell, value)
            total += reward
        assert total + _ref_choose(n, b, 0) == pytest.approx(n.beam_value(bits, width=16, depth=4, spread=1), rel=1e-6, abs=1e-6)
    stats = n.play_beam(games=2, width=8, depth=4, threads=2, seed=3, max_moves=300, spread=2)
    assert (stats["moves"] > 0).all()
    n.close()


def test_random_prefix_changes_the_games_but_stays_reproducible():
    n = nt.NTupleNet()
    n.train_choose(games=200, threads=2, seed=6, max_moves=100_000)
    plain = n.play_choose(games=4, depth=1, topk=4, threads=2, seed=3, max_moves=800)
    varied = n.play_choose(games=4, depth=1, topk=4, threads=2, seed=3, max_moves=800, prefix=20)
    again = n.play_choose(games=4, depth=1, topk=4, threads=2, seed=3, max_moves=800, prefix=20)
    assert np.array_equal(varied["scores"], again["scores"]) and (varied["moves"] > 20).all()
    assert not np.array_equal(varied["scores"], plain["scores"])   # the random tiles changed the rewards
    beam = n.play_beam(games=4, width=4, depth=3, threads=2, seed=3, max_moves=800, stride=2, prefix=20)
    beam_plain = n.play_beam(games=4, width=4, depth=3, threads=2, seed=3, max_moves=800, stride=2)
    assert (beam["moves"] > 20).all() and not np.array_equal(beam["scores"], beam_plain["scores"])
    n.close()


def test_train_choose_explore_takes_random_placements_reproducibly():
    a, b, c = nt.NTupleNet(), nt.NTupleNet(), nt.NTupleNet()
    sa = a.train_choose(games=20, threads=1, seed=3, max_moves=500)
    sb = b.train_choose(games=20, threads=1, seed=3, max_moves=500, explore=0.05)
    sc = c.train_choose(games=20, threads=1, seed=3, max_moves=500, explore=0.05)
    assert not np.array_equal(sa["scores"], sb["scores"])      # exploration changed the games
    assert np.array_equal(sb["scores"], sc["scores"])          # but deterministically
    assert (sb["moves"] > 0).all()
    a.close(); b.close(); c.close()


# --- snake-order bonus for the beam --------------------------------------------------

def test_snake_score_rewards_the_snake_layout_under_any_symmetry():
    assert nt.snake_score(0) == 0.0
    one = nt.with_cell(0, 0, 10)                                   # 1024 in a corner: head of the snake
    assert nt.snake_score(one) == pytest.approx(1024.0)
    inner = nt.with_cell(0, 5, 10)                                 # an inner cell is at best 6th on the path
    assert nt.snake_score(inner) == pytest.approx(1024.0 * 0.5 ** 5)
    chain = nt.to_bits(np.array([[17, 16, 15, 14], [10, 11, 12, 13], [9, 8, 7, 6], [2, 3, 4, 5]], dtype=np.uint8))
    expected = sum((1 << e) * 0.5 ** k for k, e in enumerate([17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]))
    assert nt.snake_score(chain) == pytest.approx(expected)
    rotated = nt.to_bits(np.rot90(nt.from_bits(chain)))
    assert nt.snake_score(rotated) == pytest.approx(expected)     # the best of the 8 symmetries is taken
    shuffled = nt.to_bits(np.array([[17, 10, 15, 6], [16, 11, 2, 13], [9, 8, 7, 14], [12, 3, 4, 5]], dtype=np.uint8))
    assert nt.snake_score(shuffled) < expected


def test_beam_snake_bonus_matches_the_reference_at_depth_one():
    n = nt.NTupleNet()
    n.train(games=200, threads=2, seed=4)
    rng = np.random.default_rng(35)
    lam = 0.7
    for _ in range(8):
        bits = nt.to_bits(rand_exps(rng, density=0.85))
        best = None
        for m in range(4):
            a, r = nt.move(bits, m)
            if a == bits:
                continue
            for i in range(16):
                if nt.cell(a, i):
                    continue
                for v in (1, 2):
                    s2 = nt.with_cell(a, i, v)
                    nxt = [r2 + n.value(a2) + lam * nt.snake_score(a2) for m2 in range(4)
                           for a2, r2 in [nt.move(s2, m2)] if a2 != s2]
                    val = r + (max(nxt) if nxt else 0.0)
                    best = val if best is None else max(best, val)
        if best is None:
            continue
        assert n.beam_value(bits, width=4096, depth=1, snake=lam) == pytest.approx(best, rel=1e-6, abs=1e-6)
    assert n.beam_value(bits, width=4096, depth=1) == n.beam_value(bits, width=4096, depth=1, snake=0.0)
    stats = n.play_beam(games=2, width=4, depth=3, threads=2, seed=3, max_moves=200, snake=1.0)
    assert (stats["moves"] > 0).all()
    n.close()


def test_ntuple_agent_accepts_a_snake_weight(tmp_path):
    n = nt.NTupleNet()
    n.train(games=50, threads=2, seed=7)
    n.save(tmp_path / "w.bin")
    n.close()
    agent = nt.NTupleAgent(weights=tmp_path / "w.bin", beam_width=4, beam_depth=3, beam_snake=1.0)
    game = Game(seed=2, spawn_mode="choose")
    for _ in range(12):
        direction, (row, col, value) = agent.choose(game)
        assert direction in game.legal_moves() and value in (2, 4)
        game.move(direction)
        game.place(row, col, value)
    assert game.score > 0


# --- late-game restarts -------------------------------------------------------------

def test_train_choose_can_start_games_from_saved_boards():
    late = nt.to_bits(np.array([[15, 14, 13, 12], [8, 9, 10, 11], [7, 6, 5, 4], [0, 0, 1, 2]], dtype=np.uint8))
    a, b, c = nt.NTupleNet(), nt.NTupleNet(), nt.NTupleNet()
    fresh = a.train_choose(games=10, threads=1, seed=3, max_moves=400)
    same = b.train_choose(games=10, threads=1, seed=3, max_moves=400, starts=[late], start_frac=0.0)
    assert np.array_equal(fresh["scores"], same["scores"])            # frac 0: nothing changes
    late_runs = c.train_choose(games=10, threads=1, seed=3, max_moves=400, starts=[late], start_frac=1.0)
    assert (late_runs["max_tiles"] >= 32768).all()                    # every game started from the late board
    assert not np.array_equal(late_runs["scores"], fresh["scores"])
    a.close(); b.close(); c.close()


def test_collect_starts_reads_the_tile_as_a_stage_boundary():
    n = nt.NTupleNet(patterns=[[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 4, 5]], tc=False)
    n.train(300, threads=1, seed=1, alpha_plain=0.1)
    one, _ = n.collect_starts(40, tile=64, depth=0, threads=2, seed=5)
    both, _ = n.collect_starts(40, tile=96, depth=0, threads=2, seed=5)       # a 64 and a 32
    assert len(one) and all(g.max() == 6 for g in one)
    assert len(both) and all(g.max() == 6 and (g == 5).any() for g in both)
    assert not all((g == 5).any() for g in one)        # the first board with a 64 need not hold a 32
    n.close()


def test_harvest_starts_collects_boards_past_a_mass_threshold(tmp_path):
    n = nt.NTupleNet()
    n.train_choose(games=100, threads=2, seed=6, max_moves=100_000)
    grids = nt.harvest_starts(n, games=2, min_mass=600, every=25, width=4, depth=3, seed=1, max_moves=1500)
    assert grids.ndim == 3 and grids.shape[1:] == (4, 4) and len(grids) > 0
    assert all(sum(1 << int(v) for v in g.reshape(16) if v) >= 600 for g in grids)   # int(): uint8 shifts overflow
    path = tmp_path / "starts.npy"
    nt.save_starts(path, grids)
    loaded = nt.load_starts(path)
    assert len(loaded) == len(grids) and all(nt.cell(b, 0) == g[0, 0] for b, g in zip(loaded, grids))
    n.close()


def test_snake_decay_is_adjustable():
    chain = nt.to_bits(np.array([[17, 16, 15, 14], [10, 11, 12, 13], [9, 8, 7, 6], [2, 3, 4, 5]], dtype=np.uint8))
    tiles = [1 << e for e in [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]]
    assert nt.snake_score(chain, decay=0.9) == pytest.approx(sum(t * 0.9 ** k for k, t in enumerate(tiles)))
    assert nt.snake_score(chain, decay=1.0) == pytest.approx(sum(tiles))
    assert nt.snake_score(chain) == pytest.approx(sum(t * 0.5 ** k for k, t in enumerate(tiles)))


def test_snake_decay_belongs_to_the_network():
    # Two players in one process (the UI's snake and ntuple-cool agents) must not
    # change each other's heuristic.
    a = nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False)
    b = nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False)
    rng = np.random.default_rng(41)
    try:
        assert a.snake_decay == b.snake_decay == nt.SNAKE_DECAY == 0.5
        a.snake_decay = 0.9
        assert a.snake_decay == 0.9 and b.snake_decay == 0.5
        a.leaf_snake = b.leaf_snake = 1.0
        for _ in range(5):
            bits = nt.to_bits(rand_exps(rng, density=0.6))
            for n in (a, b):                                           # tables are empty: the value is the bonus
                expected = max(r + nt.snake_score(after, n.snake_decay)
                               for after, r in (nt.move(bits, m) for m in range(4)) if after != bits)
                assert n.search_value(bits, depth=0) == pytest.approx(expected)
        agent = nt.NTupleAgent(net=b, snake_decay=0.75)
        assert b.snake_decay == 0.75 and agent.net is b and a.snake_decay == 0.9
    finally:
        a.close(); b.close()


def test_leaf_snake_adds_the_snake_bonus_to_random_spawn_search_leaves():
    n = nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False)
    rng = np.random.default_rng(39)
    try:
        assert n.leaf_snake == 0.0
        n.leaf_snake = 3.0
        assert n.leaf_snake == 3.0
        for _ in range(5):
            bits = nt.to_bits(rand_exps(rng, density=0.6))
            expected = max(r + 3.0 * nt.snake_score(after)
                           for after, r in (nt.move(bits, m) for m in range(4)) if after != bits)
            assert n.search_value(bits, depth=0) == pytest.approx(expected)
        n.leaf_snake = 0.0
        assert n.search_value(bits, depth=0) == pytest.approx(
            max(r for after, r in (nt.move(bits, m) for m in range(4)) if after != bits))
    finally:
        n.close()


def test_changing_the_snake_decay_flushes_cached_search_values():
    a = nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False)
    rng = np.random.default_rng(40)
    bits = nt.to_bits(rand_exps(rng, density=0.5))
    try:
        a.leaf_snake = 2.0
        at_half = a.search_value(bits, depth=1)                     # fills a's cache
        a.snake_decay = 0.9
        fresh = nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False)
        fresh.leaf_snake, fresh.snake_decay = 2.0, 0.9
        assert a.search_value(bits, depth=1) == pytest.approx(fresh.search_value(bits, depth=1))
        assert a.search_value(bits, depth=1) != pytest.approx(at_half)
        fresh.close()
    finally:
        a.close()


def test_leaf_snake_plays_random_spawn_games_better_than_rewards_alone():
    n = nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False)
    try:
        plain = n.play(games=8, depth=2, threads=4, seed=3)["scores"].mean()
        n.leaf_snake = 4.0
        snake = n.play(games=8, depth=2, threads=4, seed=3)["scores"].mean()
        assert snake > 1.5 * plain
    finally:
        n.close()


def test_beam_can_be_restricted_to_placing_twos():
    n = nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False)
    rng = np.random.default_rng(36)
    for _ in range(5):
        bits = nt.to_bits(rand_exps(rng, density=0.5))
        plan = n.beam_plan(bits, width=8, depth=6, snake=2.0, tiles=1)
        assert plan and all(v == 1 for _, _, v in plan)               # only 2s
        plan4 = n.beam_plan(bits, width=8, depth=6, snake=2.0, tiles=2)
        assert plan4 and all(v == 2 for _, _, v in plan4)             # only 4s
    stats = n.play_beam(games=1, width=8, depth=4, threads=1, seed=1, max_moves=300, snake=2.0, tiles=1)
    assert stats["moves"][0] == 300
    n.close()


def test_beam_twos_first_places_a_four_only_when_a_two_dead_ends():
    # Only left and right move. Either leaves one empty cell where a 2 fills a board
    # with no merge, while a 4 merges with its neighbour.
    exps = np.array([[1, 1, 3, 4],
                     [5, 6, 5, 2],
                     [6, 5, 6, 5],
                     [5, 6, 5, 6]])
    bits = nt.to_bits(exps)
    n = nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False)
    try:
        twos = n.beam_plan(bits, width=64, depth=3, snake=2.0, tiles=1)
        assert len(twos) == 1 and twos[0][2] == 1                       # a dead line: the board fills
        for tiles in (8, 12, 24, 28):
            plan = n.beam_plan(bits, width=64, depth=3, snake=2.0, tiles=tiles)
            assert len(plan) == 1 and plan[0][2] == 2                   # the fallback: one step, a 4
            after, _ = nt.move(bits, plan[0][0])
            board = nt.with_cell(after, plan[0][1], plan[0][2])
            assert any(nt.move(board, m)[0] != board for m in range(4))  # and the game goes on
    finally:
        n.close()


def test_beam_fewest_fours_matches_twos_first_while_twos_survive_and_ends_the_game_on_a_four():
    n = nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False)
    rng = np.random.default_rng(39)
    try:
        for _ in range(6):                                           # bit 16 only changes the fallback
            bits = nt.to_bits(rand_exps(rng, density=0.3))
            assert n.beam_plan(bits, width=16, depth=6, snake=2.0, tiles=28) == \
                n.beam_plan(bits, width=16, depth=6, snake=2.0, tiles=12)
        # one gap in a board with no two equal neighbours, before or after the only two
        # slides: the cell left over takes a 2 or a 4 without a merge, and the game ends
        last = np.array([[0, 5, 6, 7], [8, 3, 4, 3], [9, 4, 3, 4], [10, 3, 4, 3]], dtype=np.uint8)
        bits = nt.to_bits(last)
        (m, c, v), = n.beam_plan(bits, width=16, depth=6, snake=2.0, tiles=12)
        assert v == 1                                                # the plain fallback saves its 4 points
        (m2, c2, v2), = n.beam_plan(bits, width=16, depth=6, snake=2.0, tiles=28)
        assert (m2, c2, v2) == (m, c, 2)                             # same step, the tile a 4
        end = nt.with_cell(nt.move(bits, m2)[0], c2, v2)
        assert not nt.afterstates_valid(end)
    finally:
        n.close()


def test_beam_twos_first_is_the_twos_only_plan_while_twos_survive():
    n = nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False)
    rng = np.random.default_rng(38)
    try:
        for _ in range(6):
            bits = nt.to_bits(rand_exps(rng, density=0.3))
            twos = n.beam_plan(bits, width=16, depth=6, snake=2.0, tiles=1)
            assert len(twos) == 6
            assert n.beam_plan(bits, width=16, depth=6, snake=2.0, tiles=12) == twos
            assert n.beam_value(bits, width=16, depth=6, snake=2.0, tiles=12) == pytest.approx(
                n.beam_value(bits, width=16, depth=6, snake=2.0, tiles=1))
        stats = n.play_beam(games=1, width=8, depth=4, threads=1, seed=1, max_moves=300, stride=4, snake=2.0,
                            tiles=12)
        assert stats["moves"][0] == 300
    finally:
        n.close()


def test_beam_can_charge_fours_their_opportunity_cost():
    # With unlimited width the beam is the exhaustive tree, so charging 4s can only
    # lower the best line's value (a pruned beam need not be monotone in the charge).
    n = nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False)
    rng = np.random.default_rng(37)
    strict, fours_free, fours_charged = False, 0, 0
    for _ in range(8):
        bits = nt.to_bits(rand_exps(rng, density=0.6))
        v3 = n.beam_value(bits, width=4096, depth=2, snake=2.0, tiles=3)
        v7 = n.beam_value(bits, width=4096, depth=2, snake=2.0, tiles=7)
        assert v7 <= v3 + 1e-9
        strict |= v7 < v3 - 1e-9
        fours_free += sum(v == 2 for _, _, v in n.beam_plan(bits, width=4096, depth=2, snake=2.0, tiles=3))
        fours_charged += sum(v == 2 for _, _, v in n.beam_plan(bits, width=4096, depth=2, snake=2.0, tiles=7))
    assert strict and fours_charged <= fours_free                     # 4s only when they earn more than 4 points
    n.close()


# --- tile downgrading ---------------------------------------------------------

def test_downgrade_halves_every_tile_above_the_largest_missing_one():
    b = nt.to_bits([[15, 14, 12, 11], [3, 2, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    # 13 is the largest tile missing below the 15: the 15 and the 14 step down, the rest stay
    assert nt.from_bits(nt.downgrade(b)).tolist() == [[14, 13, 12, 11], [3, 2, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    full = nt.to_bits([[15, 14, 13, 12], [8, 9, 10, 11], [7, 6, 5, 4], [0, 1, 2, 3]])
    assert nt.downgrade(full) == full                  # nothing is missing, nothing to do
    assert nt.downgrade(0) == 0


def test_downgrade_keeps_every_merge_possible_and_no_other():
    rng = np.random.default_rng(11)
    for _ in range(200):
        b = nt.to_bits(rand_exps(rng, density=0.8, max_exp=15))
        d = nt.downgrade(b)
        for direction in range(4):
            moved, down_moved = nt.move(b, direction)[0], nt.move(d, direction)[0]
            assert (moved == b) == (down_moved == d)
            assert (nt.from_bits(moved) == 0).tolist() == (nt.from_bits(down_moved) == 0).tolist()


def test_downgrading_never_takes_a_board_out_of_its_stage():
    """A stage that starts at a 32768 and a 16384 is asked about those boards as they are;
    below that boundary the downgraded board is searched as before."""
    pats = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 4, 5]]
    nets = [nt.NTupleNet(patterns=pats, tc=False) for _ in range(3)]
    for i, n in enumerate(nets):
        n.train(200, threads=1, seed=i, alpha_plain=0.1)
    three = nt.NTupleNet.stack(nets, [16384, 49152])
    rng = np.random.default_rng(13)
    def board(tiles):
        g = rand_exps(rng, density=0.6, max_exp=9).ravel()
        g[: len(tiles)] = tiles
        return nt.to_bits(g.reshape(4, 4))
    alone = [board([15]) for _ in range(30)]           # 32768 and no 16384: downgrading stays in the middle stage
    pair = [board([15, 14]) for _ in range(30)]        # 32768 and 16384, no 8192: downgrading would leave the last stage
    plain = [three.best_move(b, depth=1) for b in alone + pair]
    three.downgrade = 32768
    assert [three.best_move(b, depth=1) for b in pair] == plain[30:]
    with_dg = [three.best_move(b, depth=1) for b in alone]
    three.downgrade = 0
    assert with_dg == [three.best_move(nt.downgrade(b), depth=1) for b in alone]
    for n in nets + [three]:
        n.close()


SNAKE = [0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12]


def _snake_board(k, play):
    """k tiles from 32768 down on the snake, `play` a dict cell -> exponent for the rest."""
    g = np.zeros(16, dtype=np.uint8)
    for i in range(k):
        g[SNAKE[i]] = 15 - i
    for c, v in play.items():
        g[c] = v
    return g


def _ref_endgame(g, k, memo, path=SNAKE, kvalid=15, lift=0):
    """Plain Python version of the engine's endgame rule, for the few-cell tables, along a cell path whose first
    kvalid cells are a chain of neighbours (no tables beyond); lift g: the locked tiles are 2^(15 - i - g)."""
    SNAKE = path
    n = 16 - k - lift
    key = (bytes(g), k)
    if key in memo:
        return memo[key]
    best = 0.0
    for d in range(4):
        moved, _ = nt.move(nt.to_bits(g.reshape(4, 4)), d)
        a = nt.from_bits(moved).ravel().copy()
        if (a == g).all() or any(a[SNAKE[i]] != 15 - i - lift for i in range(k - 1)):
            continue
        if a[SNAKE[k - 1]] == n + 1:
            best = 1.0
            break
        if a[SNAKE[k - 1]] != n or any(a[SNAKE[i]] > n for i in range(k, 16)):
            continue
        empty = [c for c in range(16) if a[c] == 0]
        total = 0.0
        for c in empty:
            for v, p in ((1, 0.9), (2, 0.1)):
                if v > n:
                    continue
                a[c] = v
                total += p * _ref_endgame(a.copy(), k, memo, path, kvalid, lift)
                a[c] = 0
        best = max(best, total / len(empty) if empty else 0.0)
    if k < min(kvalid, 15 - lift) and g[SNAKE[k]] == n - 1 and all(g[SNAKE[i]] <= n - 1 for i in range(k + 1, 16)):
        best = max(best, _ref_endgame(g, k + 1, memo, path, kvalid, lift))
    memo[key] = best
    return best


def test_endgame_tables_match_a_plain_python_solution():
    # one cell in play: a 2 next to the last locked 2 merges into it and wins
    assert nt.endgame_value(nt.to_bits(_snake_board(15, {12: 1}).reshape(4, 4)), 15) == 1.0
    assert nt.endgame_value(nt.to_bits(_snake_board(15, {12: 2}).reshape(4, 4)), 15) == -1.0   # a 4 is not in play
    assert nt.endgame_value(nt.to_bits(_snake_board(14, {}).reshape(4, 4)), 15) == -1.0        # not 15 locked
    rng = np.random.default_rng(14)
    memo = {}
    for k in (13, 12):
        cells = SNAKE[k:]
        for _ in range(12):
            g = _snake_board(k, {c: int(rng.integers(0, 16 - k + 1)) for c in cells})
            if not (g == 0).any():
                continue
            want = _ref_endgame(g, k, memo)
            got = nt.endgame_value(nt.to_bits(g.reshape(4, 4)), k)
            assert got == pytest.approx(want, abs=1e-6)


def test_endgame_move_is_found_in_every_orientation():
    """Fifteen tiles locked and a 2 beside the last one: the table's move is the merge that
    wins, whichever way the board is turned or mirrored."""
    n = nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False)
    n.endgame = 15
    g = _snake_board(15, {12: 1}).reshape(4, 4)
    for flip in (False, True):
        for r in range(4):
            h = np.rot90(g.T if flip else g, r)
            bits = nt.to_bits(np.ascontiguousarray(h))
            d = n.best_move(bits, depth=1)
            after, reward = nt.move(bits, d)
            assert reward == 4 and (nt.from_bits(after) == 2).sum() == 2   # the two 2s became a 4 beside the locked 4
    n.close()


def test_endgame_tables_also_serve_the_game_after_a_65536():
    """A 65536 in the corner with 16384, 8192, ... after it is played as if the corner held a 32768."""
    n = nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False)
    n.endgame = 15
    g = _snake_board(15, {12: 1})
    g[0] = 16
    bits = nt.to_bits(g.reshape(4, 4))
    d = n.best_move(bits, depth=1)
    after, reward = nt.move(bits, d)
    assert reward == 4 and nt.from_bits(after)[0, 0] == 16
    n.close()


def test_endgame_play_is_unchanged_where_no_board_qualifies():
    n = nt.NTupleNet(patterns=[[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 4, 5]], tc=False)
    n.train(200, threads=1, seed=1, alpha_plain=0.1)
    plain = n.play(4, depth=1, threads=2, seed=5)
    n.endgame = 6
    assert n.endgame == 6
    with_tables = n.play(4, depth=1, threads=2, seed=5)
    assert (plain["scores"] == with_tables["scores"]).all()       # these games never come near a 32768
    n.close()


def test_search_plays_the_downgraded_board_once_the_threshold_tile_is_there():
    n = nt.NTupleNet(tc=False)
    n.train(300, threads=1, alpha=0.1, seed=3)
    assert n.downgrade == 0
    rng = np.random.default_rng(12)
    big = [nt.to_bits(np.where(g == 15, 14, g)) | 15 for g in (rand_exps(rng, max_exp=15) for _ in range(40))]
    small = [nt.to_bits(rand_exps(rng, max_exp=13)) for _ in range(40)]
    plain = [n.best_move(b, depth=1) for b in big + small]
    n.downgrade = 32768
    assert n.downgrade == 32768
    assert [n.best_move(b, depth=1) for b in small] == plain[40:]          # no 32768 on the board: untouched
    with_dg = [n.best_move(b, depth=1) for b in big]
    n.downgrade = 0
    assert with_dg == [n.best_move(nt.downgrade(b), depth=1) for b in big]
    # Raise one move's afterstate on the downgraded board and lower the same move's on
    # the real board: the entries they share cancel, the ones that see the big tiles do not.
    b = nt.to_bits([[15, 14, 12, 3], [2, 5, 1, 4], [1, 3, 2, 1], [0, 0, 1, 2]])
    other = next(m for m in nt.afterstates_valid(b) if m != n.best_move(b, depth=1))
    n.update(nt.move(nt.downgrade(b), other)[0], 1e6, alpha=1.0)
    n.update(nt.move(b, other)[0], -1e6, alpha=1.0)
    assert n.best_move(b, depth=1) != other
    n.downgrade = 32768
    assert n.best_move(b, depth=1) == other
    n.close()


def _trained_small_net():
    n = nt.NTupleNet(patterns=[[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 4, 5]], tc=False)
    n.train(games=400, threads=2, seed=2, alpha_plain=0.1)
    return n


def _some_boards(count, seed=0):
    """Boards from random play, a few moves apart."""
    boards, b, k = [], nt.spawn(nt.spawn(0, seed), seed + 1), seed + 2
    while len(boards) < count:
        valid = nt.afterstates_valid(b)
        if not valid:
            b = nt.spawn(nt.spawn(0, k), k + 1)
            continue
        b = nt.spawn(nt.move(b, valid[k % len(valid)])[0], k)
        k += 1
        if k % 7 == 0:
            boards.append(b)
    return boards


def test_search_play_does_not_depend_on_threads_or_table_size():
    n = _trained_small_net()
    one = n.play(games=6, depth=2, threads=1, seed=4)
    three = n.play(games=6, depth=2, threads=3, seed=4)
    assert (one["scores"] == three["scores"]).all() and (one["moves"] == three["moves"]).all()
    n.tt_bits = 10                                               # a table far too small for the search
    assert n.tt_bits == 10
    tiny = n.play(games=6, depth=2, threads=2, seed=4)
    assert (one["scores"] == tiny["scores"]).all()
    for b in _some_boards(20):
        v = n.search_value(b, depth=2)
        n.tt_bits = 14
        assert n.search_value(b, depth=2) == v
        n.tt_bits = 10


def test_depth_caps_limit_the_search_by_empty_cells():
    n = _trained_small_net()
    assert n.depth_caps == [0] * 17
    boards = _some_boards(40, seed=5)
    deep = [n.best_move(b, depth=3) for b in boards]
    shallow = [n.best_move(b, depth=1) for b in boards]
    assert deep != shallow                                       # the depth matters on these boards
    n.depth_caps = [1]
    assert n.depth_caps == [1] * 17
    assert [n.best_move(b, depth=3) for b in boards] == shallow
    n.depth_caps = [3, 3, 3, 3, 3, 3, 1]                         # deep with 0-5 empty cells, shallow from 6
    empties = [sum(nt.cell(b, i) == 0 for i in range(16)) for b in boards]
    assert min(empties) <= 5 < max(empties)
    assert [n.best_move(b, depth=3) for b in boards] == [d if e <= 5 else s for d, s, e in zip(deep, shallow, empties)]
    n.depth_caps = []
    assert [n.best_move(b, depth=3) for b in boards] == deep


def test_expected_target_averages_over_every_tile_that_can_appear():
    n = _trained_small_net()
    for b in _some_boards(12, seed=7):
        valid = nt.afterstates_valid(b)
        if not valid:
            continue
        after, _ = nt.move(b, valid[0])
        empty = [i for i in range(16) if nt.cell(after, i) == 0]
        total = 0.0
        for i in empty:
            for exp, prob in ((1, 0.9), (2, 0.1)):
                s = nt.with_cell(after, i, exp)
                options = [r + n.value(a) for a, r in (nt.move(s, d) for d in range(4)) if a != s]
                total += prob * (max(options) if options else 0.0)
        assert n.expected_target(after) == pytest.approx(total / len(empty), rel=1e-5)
    n.close()


def test_training_on_the_expected_target_learns_and_is_off_by_default():
    pats = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 4, 5]]
    plain, again, expect = (nt.NTupleNet(patterns=pats, tc=False) for _ in range(3))
    assert plain.expect_target is False
    a = plain.train(games=300, threads=1, seed=2, alpha_plain=0.1)
    b = again.train(games=300, threads=1, seed=2, alpha_plain=0.1)
    assert (a["scores"] == b["scores"]).all()                    # the default path is untouched
    expect.expect_target = True
    assert expect.expect_target is True
    c = expect.train(games=300, threads=1, seed=2, alpha_plain=0.1)
    assert not (a["scores"] == c["scores"]).all()                # a different learner
    assert c["scores"][-100:].mean() > 1.4 * c["scores"][:20].mean() # and it learns
    for n in (plain, again, expect):
        n.close()


_PREBUILT_CHECK = r"""
import sys
import numpy as np
from game2048 import ntuple as nt
SNAKE = [0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12]
rng = np.random.default_rng(0)
boards = []
for k in range(12, 16):
    for _ in range(300):
        g = np.zeros(16, dtype=np.uint8)
        for i in range(k):
            g[SNAKE[i]] = 15 - i
        for i in range(k, 16):
            g[SNAKE[i]] = 0 if rng.random() < 0.3 else rng.integers(1, 16 - k + 1)
        boards.append((nt.to_bits(g.reshape(4, 4)), k))
before = [nt.endgame_value(b, k) for b, k in boards]          # solved on demand
assert nt.load_endgame(sys.argv[1]) == 12
after = [nt.endgame_value(b, k) for b, k in boards]           # read from the files
assert all((x < 0) == (y < 0) for x, y in zip(before, after))
assert max(abs(x - y) for x, y in zip(before, after)) < 1e-4
assert sum(x > 0 for x in after) > 100
with __import__("pytest").raises(ValueError):
    nt.load_endgame(sys.argv[1])                               # once per process
print("ok")
"""


def test_endgame_tables_built_in_advance_match_the_solver(tmp_path):
    """Tables from examples/build_endgame.py (12 to 15 locked: exact, no caps) give the
    values the engine solves on demand. Run in a fresh process: loading is once per process."""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    out = tmp_path / "tables"
    subprocess.run([sys.executable, str(root / "examples" / "build_endgame.py"), "--kmin", "12", "--threads", "2",
                    "--out", str(out)], check=True, capture_output=True)
    assert (out / "caps.txt").read_text().split() == ["12", "0,1,2,3,3,4,4,5,5,5,5", "6", "2"]
    r = subprocess.run([sys.executable, "-c", _PREBUILT_CHECK, str(out)], capture_output=True, text=True, cwd=root)
    assert r.stdout.strip() == "ok", r.stdout + r.stderr


def test_play_boards_from_start_boards():
    """play_boards plays each start board to the end: the same games whatever the thread count,
    nothing from a board with no move, and a fresh board plays a whole game."""
    net = nt.NTupleNet(patterns=[[0, 1, 2, 3], [4, 5, 6, 7]], tc=False)
    dead = np.array([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]], dtype=np.uint8)
    start = np.zeros((4, 4), dtype=np.uint8); start[0, 0] = 1; start[3, 3] = 2
    boards = np.stack([dead, start, start, start])
    a = net.play_boards(boards, depth=0, threads=1, seed=5)
    b = net.play_boards(boards, depth=0, threads=3, seed=5)
    for k in ("max_tile", "score", "moves"):
        assert (a[k] == b[k]).all()
    assert a["moves"][0] == 0 and a["score"][0] == 0 and a["max_tile"][0] == 4
    assert (a["moves"][1:] > 20).all() and (a["max_tile"][1:] >= 16).all()
    assert len(set(a["score"][1:].tolist())) > 1                 # each game draws its own tiles
    assert a["seen"][0] == (1 << 1) | (1 << 2)                    # the dead board: only 2s and 4s
    assert all((s >> 1) & 1 and (int(t).bit_length() - 1) <= 31 and (s >> (int(t).bit_length() - 1)) & 1
               for s, t in zip(a["seen"][1:], a["max_tile"][1:]))  # the largest tile was seen


def test_downgrade_repeats_after_a_65536():
    """After a 65536 one downgrading step leaves a 32768 on top; the search downgrades again and plays the
    board as the 16384 board it saw before the 65536. A board whose largest tile is 32768 still gets one step."""
    net = nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False)
    net.downgrade = 32768
    g = np.zeros(16, dtype=np.uint8)
    g[0], g[1], g[2], g[5] = 16, 14, 13, 3                       # 65536 16384 8192 8
    want = g.copy(); want[0], want[1], want[2] = 14, 13, 12        # 16384 8192 4096 8
    assert (nt.from_bits(net.root_board(nt.to_bits(g.reshape(4, 4)))).ravel() == want).all()
    g[0] = 15                                                      # 32768 16384 8192 8: one step only
    want[0], want[1], want[2] = 14, 13, 12
    assert (nt.from_bits(net.root_board(nt.to_bits(g.reshape(4, 4)))).ravel() == want).all()
    net.downgrade = 0
    assert net.root_board(nt.to_bits(g.reshape(4, 4))) == nt.to_bits(g.reshape(4, 4))


_FAMILY_CHECK = r"""
import json, sys
import numpy as np
from game2048 import ntuple as nt
snake_dir, block_dir, boards = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
assert nt.load_endgame(snake_dir) == 12 and nt.load_endgame(block_dir) == 12
assert nt.lib().nt_endgame_families() == 2
out = [nt.endgame_value(nt.to_bits(np.array(g, dtype=np.uint8).reshape(4, 4)), k, 1) for g, k in boards]
net = nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False)
r = net.reach(np.array([boards[0][0]], dtype=np.uint8).reshape(1, 4, 4), k=12, families=2)
g = np.zeros(16, dtype=np.uint8)
for i in range(13):
    g[nt.BLOCK_PATH[i]] = 15 - i
beyond = nt.endgame_value(nt.to_bits(g.reshape(4, 4)), 13, 1)
print(json.dumps({"values": out, "block_reached": int(r["reached"][0]), "beyond": beyond}))
"""


def test_endgame_tables_of_a_second_formation(tmp_path):
    """Tables built along the 2x3-block path load as a second formation beside the snake, give the
    values of a plain Python solution along that path, and count for reach with their bit."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for name in ("snake", "block"):
        subprocess.run([sys.executable, str(root / "examples" / "build_endgame.py"), "--kmin", "12", "--threads", "2",
                        "--path", name, "--out", str(tmp_path / name)], check=True, capture_output=True)
    assert (tmp_path / "block" / "caps.txt").read_text().split()[4] == ",".join(map(str, nt.BLOCK_PATH))
    rng = np.random.default_rng(21)
    boards, want, memo = [], [], {}
    while len(boards) < 16:
        k = 12                       # the block path's cells are neighbours up to 12 locked (cell 3 to 15 is not)
        g = np.zeros(16, dtype=np.uint8)
        for i in range(k):
            g[nt.BLOCK_PATH[i]] = 15 - i
        for c in nt.BLOCK_PATH[k:]:
            g[c] = int(rng.integers(0, 16 - k + 1))
        if not (g == 0).any():
            continue
        boards.append((g.tolist(), k))
        want.append(_ref_endgame(g, k, memo, nt.BLOCK_PATH, 12))
    r = subprocess.run([sys.executable, "-c", _FAMILY_CHECK, str(tmp_path / "snake"), str(tmp_path / "block"),
                        json.dumps(boards)], capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["values"] == pytest.approx(want, abs=1e-4)
    assert got["block_reached"] == 1
    assert got["beyond"] == -1.0     # 13 locked along the block path is past its chain: no table



def test_search_never_picks_a_move_only_its_downgraded_twin_allows():
    """Two 131072s cannot merge (the largest tile), but downgraded they look like two smaller tiles that can:
    the search must only take moves that are legal on the real board, or play_boards would loop forever."""
    net = nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False)
    net.downgrade = 32768
    g = np.array([[17, 17, 1, 2], [3, 4, 2, 1], [4, 3, 1, 2], [3, 4, 2, 1]], dtype=np.uint8)
    assert net.best_move(nt.to_bits(g), depth=1) == -1
    r = net.play_boards(g.reshape(1, 4, 4), depth=1, threads=1, seed=1)
    assert r["moves"][0] == 0


def test_endgame_tables_treat_finished_tiles_as_boulders():
    """After a 65536 and a 32768 the ladder being built starts lower: 65536 32768 | 8192 4096 ... . The finished
    tiles never move and nothing in play can merge with them, so the board plays exactly like the table's own
    board (32768 16384 | 8192 4096 ...): same moves. Equal neighbours on the path are not boulders."""
    rng = np.random.default_rng(9)
    net = nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False)
    net.endgame = 13
    for _ in range(12):
        real = np.zeros(16, dtype=np.uint8)
        table = np.zeros(16, dtype=np.uint8)
        for i in range(13):
            table[SNAKE[i]] = 15 - i
            real[SNAKE[i]] = 15 - i
        real[SNAKE[0]], real[SNAKE[1]] = 17, 16                      # 131072 65536 as boulders
        for c in SNAKE[13:]:
            v = int(rng.integers(0, 4))
            real[c] = table[c] = v
        if not (real == 0).any():
            continue
        a = net.best_move(nt.to_bits(real.reshape(4, 4)), depth=1)
        b = net.best_move(nt.to_bits(table.reshape(4, 4)), depth=1)
        assert a == b
        r = net.reach(real.reshape(1, 4, 4), k=13, depth=0, threads=1, seed=1)
        assert r["reached"][0] == 1 and r["moves"][0] == 0
    dup = np.zeros(16, dtype=np.uint8)
    for i in range(13):
        dup[SNAKE[i]] = 15 - i
    dup[SNAKE[1]] = 15                                              # 32768 32768: they could merge
    r = net.reach(dup.reshape(1, 4, 4), k=13, depth=0, threads=1, seed=1)
    assert r["moves"][0] > 0 or r["reached"][0] == 0


_LIFT_CHECK = r"""
import json, sys
import numpy as np
from game2048 import ntuple as nt
d, boards = sys.argv[1], json.loads(sys.argv[2])
assert nt.load_endgame(d) == 11
out = [nt.endgame_value(nt.to_bits(np.array(g, dtype=np.uint8).reshape(4, 4)), k, 1) for g, k in boards]
# a real board with a gap: 32768 16384 | 4096 2048 ... (8192 being built) is a position of the lifted tables
net = nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False)
g = np.zeros(16, dtype=np.uint8)
top = [15, 14] + [12 - i for i in range(9)]                      # 11 locked: 32768 16384 | 4096 ... 8
for i, e in enumerate(top):
    g[nt.SNAKE_PATH[i]] = e
g[nt.SNAKE_PATH[13]] = 1
r = net.reach(g.reshape(1, 4, 4), k=11, depth=0, threads=1, seed=1, families=2)
print(json.dumps({"values": out, "gap_reached": int(r["reached"][0] and r["moves"][0] == 0)}))
"""


_LIFT_PRIORITY_CHECK = r"""
import json, sys
import numpy as np
from game2048 import ntuple as nt
d = sys.argv[1]
net = nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False)
net.endgame = 12
rng = np.random.default_rng(5)
boards = []
while len(boards) < 40:
    g = np.zeros(16, dtype=np.uint8)
    for i in range(12):
        g[nt.SNAKE_PATH[i]] = 15 - i                               # a whole ladder 32768 ... 16
    for c in nt.SNAKE_PATH[13:]:
        g[c] = int(rng.integers(0, 4))
    g[nt.SNAKE_PATH[12]] = 2                                       # a 4 next: also a lift-1 position (13 locked)
    if (g == 0).any():
        boards.append(g.reshape(4, 4))
before = [net.best_move(nt.to_bits(g), depth=1) for g in boards]
assert nt.load_endgame(d) == 11
after = [net.best_move(nt.to_bits(g), depth=1) for g in boards]
print(json.dumps({"same": sum(a == b for a, b in zip(before, after)), "n": len(boards)}))
"""


def test_sub_goal_tables_with_a_lift(tmp_path):
    """Tables with lift 1 (the last locked tile one size below the usual one) give the values of the plain
    Python solution, and a real board with a gap in its ladder is a position of them (the tiles above the gap
    read as boulders)."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    subprocess.run([sys.executable, str(root / "examples" / "build_endgame.py"), "--kmin", "11", "--threads", "2",
                    "--lift", "1", "--out", str(tmp_path / "lift1")], check=True, capture_output=True)
    assert (tmp_path / "lift1" / "caps.txt").read_text().split()[5] == "1"
    rng = np.random.default_rng(31)
    boards, want, memo = [], [], {}
    while len(boards) < 12:
        k = int(rng.integers(11, 13))
        g = np.zeros(16, dtype=np.uint8)
        for i in range(k):
            g[SNAKE[i]] = 14 - i
        for c in SNAKE[k:]:
            g[c] = int(rng.integers(0, 16 - k - 1 + 1))
        if not (g == 0).any():
            continue
        boards.append((g.tolist(), k))
        want.append(_ref_endgame(g, k, memo, SNAKE, 15, 1))
    r = subprocess.run([sys.executable, "-c", _LIFT_CHECK, str(tmp_path / "lift1"), json.dumps(boards)],
                       capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["values"] == pytest.approx(want, abs=1e-4)
    assert got["gap_reached"] == 1
    # a board the snake tables hold plays the snake's move: the lift-1 chance (of the next tile alone) never
    # outbids it
    r = subprocess.run([sys.executable, "-c", _LIFT_PRIORITY_CHECK, str(tmp_path / "lift1")],
                       capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["same"] == got["n"]




def test_deep_window_searches_deeper_only_when_the_board_holds_the_tiles():
    """net.deep = (depth, tiles): boards holding every listed tile are searched at that depth, others as asked."""
    rng = np.random.default_rng(12)
    net = nt.NTupleNet(patterns=[[0, 1, 2, 3], [4, 5, 6, 7]], tc=False)
    net.deep = (3, [2048, 1024])
    assert net.deep == (3, [1024, 2048])
    differ = 0
    for i in range(60):
        g = rng.integers(0, 8, size=16).astype(np.uint8)
        g[rng.integers(0, 16)] = 0
        if i % 2:
            g[0], g[1] = 11, 10
        b = nt.to_bits(g.reshape(4, 4))
        has = {11, 10} <= set(int(x) for x in g)
        got = net.best_move(b, depth=1)
        net.deep = None
        shallow, deep = net.best_move(b, depth=1), net.best_move(b, depth=3)
        net.deep = (3, [2048, 1024])
        assert got == (deep if has else shallow)
        differ += has and deep != shallow
    assert differ > 0
    net.deep = None
    assert net.deep == (0, [])


def _ref_wobble(g, k, o, memo):
    """Plain Python version of the builder's wobble rule (uncapped tables): the locked tiles after the last full
    row (the block) may slide along their row; the block sits o cells beyond home; a win only at home."""
    key = (bytes(g), k, o)
    if key in memo:
        return memo[key]
    rs = (k - 1) // 4 * 4
    L, n = k - rs, 16 - k

    def block(a):
        for p in range(4 - L + 1):
            if all(a[SNAKE[rs + p + i]] == 15 - (rs + i) for i in range(L - 1)):
                last = a[SNAKE[rs + p + L - 1]]
                if last in (n, n + 1):
                    return p, last == n + 1
        return -1, False

    def free(p):
        return [SNAKE[q] for q in range(rs, 16) if not rs + p <= q < rs + p + L]

    best = 0.0
    for d in range(4):
        moved, _ = nt.move(nt.to_bits(g.reshape(4, 4)), d)
        a = nt.from_bits(moved).ravel().copy()
        if (a == g).all() or any(a[SNAKE[i]] != 15 - i for i in range(rs)):
            continue
        p, won = block(a)
        if p < 0:
            continue
        if won:
            if p == 0:
                best = 1.0
                break
            continue
        if any(a[c] > n for c in free(p)):
            continue
        empty = [c for c in range(16) if a[c] == 0]
        total = 0.0
        for c in empty:
            for v, pr in ((1, 0.9), (2, 0.1)):
                if v > n:
                    continue
                a[c] = v
                total += pr * _ref_wobble(a.copy(), k, p, memo)
                a[c] = 0
        best = max(best, total / len(empty) if empty else 0.0)
    q = rs + o + L
    if k < 15 and (q < rs + 4 if L < 4 else True) and g[SNAKE[q]] == n - 1 \
            and all(g[c] <= n - 1 for c in free(o) if c != SNAKE[q]):
        best = max(best, _ref_wobble(g, k + 1, o if L < 4 else 0, memo))
    memo[key] = best
    return best


_WOBBLE_CHECK = r"""
import json, sys
import numpy as np
from game2048 import ntuple as nt
d, boards = sys.argv[1], json.loads(sys.argv[2])
assert nt.load_endgame(d) == 12
f = nt.lib().nt_endgame_families() - 1
out = [nt.endgame_value(nt.to_bits(np.array(g, dtype=np.uint8).reshape(4, 4)), k, f) for g, k, _ in boards]
net = nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False)
net.endgame = 12
moves = [net.best_move(nt.to_bits(np.array(g, dtype=np.uint8).reshape(4, 4)), depth=1) for g, k, _ in boards]
print(json.dumps({"values": out, "moves": moves}))
"""


def test_wobble_tables_match_a_plain_python_solution(tmp_path):
    """Tables built with --wobble (the last row's locked tiles may slide along it) give the plain Python values,
    with the block at home or slid, and are at least the plain tables' values at home."""
    import ctypes
    import os
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for wob, d in ((True, "wob"), (False, "plain")):
        subprocess.run([sys.executable, str(root / "examples" / "build_endgame.py"), "--kmin", "12", "--threads", "2",
                        "--out", str(tmp_path / d)] + (["--wobble"] if wob else []), check=True, capture_output=True)
    assert (tmp_path / "wob" / "caps.txt").read_text().split()[-1] == "wobble"
    libs = []
    for d in ("wob", "plain"):
        so = tmp_path / f"egb_{d}.so"
        subprocess.run([os.environ.get("CC", "cc"), "-O2", "-shared", "-fPIC", "-o", str(so),
                        str(root / "game2048" / "native" / "endgame_build.c")], check=True)
        L = ctypes.CDLL(str(so))
        L.tb2_value.restype = ctypes.c_float
        L.tb2_set_wobble(int(d == "wob"))
        drops = (ctypes.c_int * 1)(0)
        assert L.tb2_load(str(tmp_path / d).encode(), 12, drops, 1, 6) == 0
        libs.append(L)
    import json
    rng = np.random.default_rng(36)
    memo, n_checked, slid, boards = {}, 0, 0, []
    while n_checked < 30:
        k = int(rng.integers(13, 15))
        rs, L_ = (k - 1) // 4 * 4, k - (k - 1) // 4 * 4
        o = int(rng.integers(0, 4 - L_ + 1))
        g = np.zeros(16, dtype=np.uint8)
        for i in range(rs):
            g[SNAKE[i]] = 15 - i
        for i in range(L_):
            g[SNAKE[rs + o + i]] = 15 - (rs + i)
        for q in range(rs, 16):
            if not rs + o <= q < rs + o + L_:
                g[SNAKE[q]] = int(rng.integers(0, 16 - k + 1))
        if not (g == 0).any():
            continue
        b = (ctypes.c_uint8 * 16)(*g.tolist())
        got = libs[0].tb2_value(b, k)
        if got < 0:                       # another reading of the board (a smaller offset) won or does not fit
            continue
        assert got == pytest.approx(_ref_wobble(g, k, o, memo), abs=1e-4)
        if o == 0:
            assert got >= libs[1].tb2_value(b, k) - 1e-4
        slid += o > 0
        n_checked += 1
        boards.append((g.tolist(), k, float(got)))
    assert slid > 0
    # the engine reads the same values (its own lookup, block found at any offset) and plays with them
    r = subprocess.run([sys.executable, "-c", _WOBBLE_CHECK, str(tmp_path / "wob"), json.dumps(boards)],
                       capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["values"] == pytest.approx([v for _, _, v in boards], abs=1e-6)
    assert all(m >= 0 for m, (_, _, v) in zip(got["moves"], boards) if v > 0)   # a chance: a table move


_BONUS_CHECK = r"""
import json, sys
import numpy as np
from game2048 import ntuple as nt
assert nt.load_endgame(sys.argv[1]) == 12
net = nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False)
net.endgame = 12                                  # 11 rungs are not a table position: the search plays
g = np.zeros(16, dtype=np.uint8)
for i in range(11):
    g[nt.SNAKE_PATH[i]] = 15 - i                  # 32768 ... 32 on the snake's first 11 cells
g[11] = 3
g[12:16] = 3                                      # UP merges the 8s on cells 11 and 15: the 16 on cell 11 is the
                                                  # 12th rung; a sideways move scores 32 (UP only 16), no table
b = nt.to_bits(g.reshape(4, 4))
plain = net.best_move(b, depth=0)            # greedy: the move and its points only
net.endgame_bonus = 1e9
print(json.dumps({"plain": plain, "bonus": net.best_move(b, depth=0), "get": net.endgame_bonus}))
"""


def test_endgame_bonus_steers_a_root_move_into_a_table(tmp_path):
    """With endgame_bonus a root move whose afterstate is a table position (tables built in advance) gains the
    bonus times the table's chance: a huge bonus makes the search take the move into the table."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    subprocess.run([sys.executable, str(root / "examples" / "build_endgame.py"), "--kmin", "12", "--threads", "2",
                    "--out", str(tmp_path / "t")], check=True, capture_output=True)
    r = subprocess.run([sys.executable, "-c", _BONUS_CHECK, str(tmp_path / "t")], capture_output=True, text=True,
                       cwd=root)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["plain"] in (1, 3)                 # without the bonus: the 32 points
    assert got["bonus"] == 0 and got["get"] == 1e9


# --- block-tier tables ----------------------------------------------------------

_BLK_CELLS = {"b6": [0, 1, 2, 4, 5, 6], "b5": [0, 1, 2, 4, 5]}
_BLK_FREE = {"b6": [3, 7, 8, 9, 10, 11, 12, 13, 14, 15], "b5": [3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}
_BLK_CAP = {"b6": dict(zip(_BLK_FREE["b6"], [1, 2, 1, 2, 3, 3, 3, 3, 3, 3])),       # free cell: cap = a - this
            "b5": dict(zip(_BLK_FREE["b5"], [2, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3]))}


def _blk_anchor(sh, b):
    """The block's anchor (0: not a regular block), by the rule in block_build.c."""
    if not b[0] or not b[1] or b[0] == b[1] or b[1] == b[2] or b[0] == b[4] or b[1] == b[5]:
        return 0
    if sh == "b6":
        return 0 if b[4] == b[5] or b[5] == b[6] or b[2] <= b[6] else min(b[4], b[5], b[6])
    return 0 if b[5] == b[2] or b[4] <= b[5] else min(b[5], b[2])


def _blk_goal(sh, b, a):
    if sh == "b6":
        x, y, z = a - 1, a - 2, a - 3
        return ((b[8] == x and b[9] == y and ((b[10] == z and z in (b[14], b[11])) or
                                               (b[13] == z and z in (b[12], b[14])))) or
                (b[3] == x and b[7] == y and b[11] == z and z in (b[10], b[15])))
    t = a - 1
    return b[6] == t and ((b[7] == t - 1 and t - 1 in (b[3], b[11])) or (b[3] == t - 1 and b[7] == t - 2 == b[11]) or
                          (b[10] == t - 1 and t - 1 in (b[9], b[14], b[11])))


def _blk_move(b, d):
    """Plain 2048 move on 16 exponents (0 up, 1 right, 2 down, 3 left)."""
    out = list(b)
    for i in range(4):
        cells = [[4 * j + i, 4 * i + 3 - j, 4 * (3 - j) + i, 4 * i + j][d] for j in range(4)]
        t, line = [b[c] for c in cells if b[c]], []
        while t:
            if len(t) > 1 and t[0] == t[1]:
                line.append(t[0] + 1)
                t = t[2:]
            else:
                line.append(t.pop(0))
        for c, v in zip(cells, line + [0] * (4 - len(line))):
            out[c] = v
    return out


def _ref_blk_after(m, sh, a, memo):
    """Chance of an afterstate m (the block unmoved) on the real board: the rule written out plainly."""
    if any(m[c] > a - k for c, k in _BLK_CAP[sh].items()):
        return 0.0
    if _blk_goal(sh, m, a):
        return 1.0
    empty, tot = [c for c in _BLK_FREE[sh] if m[c] == 0], 0.0
    for c in empty:
        for v, p in ((1, 0.9), (2, 0.1)):
            if v <= a - _BLK_CAP[sh][c]:
                tot += p * _ref_blk(m[:c] + [v] + m[c + 1:], sh, a, memo)
    return tot / len(empty)


def _ref_blk_moves(b, sh, a, memo):
    """Each move's chance (None: not allowed, the board does not change or the block moves)."""
    out = []
    for d in range(4):
        m = _blk_move(b, d)
        ok = m != b and all(m[c] == b[c] for c in _BLK_CELLS[sh])
        out.append(_ref_blk_after(m, sh, a, memo) if ok else None)
    return out


def _ref_blk(b, sh, a, memo):
    key = (sh, a, tuple(b))
    if key not in memo:
        memo[key] = max([v for v in _ref_blk_moves(b, sh, a, memo) if v is not None], default=0.0)
    return memo[key]


def _blk_board(sh, anchor, a, rng, fill=0.6):
    """A random board of table (sh, a): a regular block with this anchor (its other tiles any size, small ones
    too), free tiles under the caps of a."""
    while True:
        b = [int(rng.integers(1, anchor + 6)) if c in _BLK_CELLS[sh] else 0 for c in range(16)]
        if _blk_anchor(sh, b) == anchor:
            break
    for c, k in _BLK_CAP[sh].items():
        b[c] = int(rng.integers(0, a - k + 1)) if rng.random() < fill else 0
    return b


_BLOCK_CHECK = r"""
import json, sys
import numpy as np
from game2048 import ntuple as nt
dirs, boards, turned = sys.argv[1].split(","), json.loads(sys.argv[2]), json.loads(sys.argv[3])
loaded = [nt.load_block_tables(d) for d in dirs]
bits = lambda g: nt.to_bits(np.array(g, dtype=np.uint8).reshape(4, 4))
values = [nt.block_value(bits(g), sh, a) for g, sh, a in boards]
net = nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False)
assert not net.block_tier
net.block_tier = True
moves = [net.best_move(bits(g), depth=1) for g in turned]
print(json.dumps({"loaded": loaded, "values": values, "moves": moves, "count": nt.block_tier_moves(),
                  "tier": net.block_tier}))
"""


def test_block_tables_match_a_plain_python_solution(tmp_path):
    """Block-tier tables (examples/build_endgame.py --block) give the plain Python values on random boards whose
    block holds any regular tiles (anchors at and above the table's), the engine reads the same values, and the
    tier's move, on the board turned any way, is a best move of the table."""
    import ctypes
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for sh, anchors in (("b6", "4,5"), ("b5", "4")):
        subprocess.run([sys.executable, str(root / "examples" / "build_endgame.py"), "--block", sh, "--anchors",
                        anchors, "--threads", "2", "--out", str(tmp_path / sh)], check=True, capture_output=True)
    assert (tmp_path / "b6" / "block.txt").read_text().split() == ["b6", "4,5", "3,7,8,9,10,11,12,13,14,15",
                                                                  "1,2,1,2,3,3,3,3,3,3"]
    so = tmp_path / "bb.so"
    subprocess.run([os.environ.get("CC", "cc"), "-O2", "-shared", "-fPIC", "-o", str(so),
                    str(root / "game2048" / "native" / "block_build.c")], check=True)
    L = ctypes.CDLL(str(so))
    L.tbb_value.restype = ctypes.c_float
    tables = {("b6", 4): 0, ("b6", 5): 0, ("b5", 4): 1}
    for (sh, a), i in tables.items():
        assert L.tbb_load(str(tmp_path / sh).encode(), i, a) == 0
    rng = np.random.default_rng(41)
    memo, boards, turned, inside = {}, [], [], 0
    for (sh, a), i in tables.items():
        for n in range(16 if a == 4 else 6):
            b = _blk_board(sh, a + int(rng.integers(0, 3)), a, rng, fill=0.6 if a == 4 else 0.8)
            got = L.tbb_value((ctypes.c_uint8 * 16)(*b), i, a)
            want = _ref_blk(b, sh, a, memo)
            assert got == pytest.approx(want, abs=1e-4)
            inside += 0.0 < want < 1.0
            if a == 4:                                  # another block, the same free tiles: the same chance
                other = _blk_board(sh, a + int(rng.integers(0, 3)), a, rng)
                other = [other[c] if c in _BLK_CELLS[sh] else b[c] for c in range(16)]
                assert _ref_blk(other, sh, a, memo) == pytest.approx(want, abs=1e-9)
            boards.append((b, sh, a, float(got)))
            # the tier's move (b6 a 5 and b5 a 4 are each the only reading of their boards with these tables)
            if (sh, a) != ("b6", 4) and not _blk_goal(sh, b, a) and want > 0:
                s = int(rng.integers(0, 8))
                real = [0] * 16
                for j in range(16):
                    real[nt.sym_cell(j, s)] = b[j]
                turned.append((real, b, sh, a, s, want))
    assert inside >= 10 and len(turned) >= 8
    r = subprocess.run([sys.executable, "-c", _BLOCK_CHECK, f"{tmp_path / 'b6'},{tmp_path / 'b5'}",
                        json.dumps([(b, sh, a) for b, sh, a, _ in boards]), json.dumps([t[0] for t in turned])],
                       capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["loaded"] == [[4, 5], [4]] and got["tier"]
    assert got["values"] == pytest.approx([v for *_, v in boards], abs=1e-7)    # the same u16 entries
    assert got["count"] == len(turned)
    for (real, b, sh, a, s, want), d in zip(turned, got["moves"]):
        after = nt.from_bits(nt.move(nt.to_bits(np.array(real, dtype=np.uint8).reshape(4, 4)), d)[0]).ravel()
        m = [int(after[nt.sym_cell(j, s)]) for j in range(16)]             # back in the table's frame
        assert all(m[c] == b[c] for c in _BLK_CELLS[sh])
        assert _ref_blk_after(m, sh, a, memo) == pytest.approx(want, abs=1e-3)


_BLOCK_OFF_CHECK = r"""
import json, sys
from game2048 import ntuple as nt
net = nt.NTupleNet(patterns=[[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 4, 5]], tc=False)
net.train(200, threads=1, seed=1, alpha_plain=0.1)
play = lambda: net.play(6, depth=1, threads=2, seed=5)["scores"].tolist()
plain = play()
net.block_tier = True                             # on, no block tables loaded
empty = play()
nt.load_block_tables(sys.argv[1])
net.block_tier = False                            # tables loaded, tier off
off = play()
net.block_tier = True
on = play()
print(json.dumps({"plain": plain, "empty": empty, "off": off, "on": on, "count": nt.block_tier_moves()}))
"""


def test_block_tier_off_or_without_tables_leaves_play_unchanged(tmp_path):
    """Games are the same with the tier off (tables loaded) or on without tables; on with tables, the tier plays."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    subprocess.run([sys.executable, str(root / "examples" / "build_endgame.py"), "--block", "b6", "--anchors",
                    "4,5,6", "--threads", "2", "--out", str(tmp_path / "b6")], check=True, capture_output=True)
    r = subprocess.run([sys.executable, "-c", _BLOCK_OFF_CHECK, str(tmp_path / "b6")], capture_output=True,
                       text=True, cwd=root)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["empty"] == got["plain"] and got["off"] == got["plain"]
    assert got["count"] > 0                       # small blocks turn up early in every game


# --- macroxue's heuristic for the late game ------------------------------------

def _ref_heur(bits):
    """macroxue's node.h (BuildScoreMap, Evaluate without -I), written out plainly."""
    def line(cells):
        t = [c << c for c in cells]
        s = t[0]
        for a, b in zip(t, t[1:]):
            s += a + b if a >= b else (a - b) * 12
            s += a if a == b else 0
        return s
    g = [nt.cell(bits, i) for i in range(16)]
    return sum(line(g[4 * r:4 * r + 4]) for r in range(4)) + sum(line(g[c::4]) for c in range(4))


def _ref_orient(bits):
    """Largest tile nearest the top left, then the larger first row, first column, rest row by row."""
    order = [0, 1, 2, 3, 4, 8, 12, 5, 6, 7, 9, 10, 11, 13, 14, 15]
    best = None
    for s in range(8):
        o = [nt.cell(bits, nt.sym_cell(i, s)) for i in range(16)]
        near = min(i // 4 + i % 4 for i in range(16) if o[i] == max(o))
        key = [-near] + [o[i] for i in order]
        if best is None or key > best[0]:
            best = (key, s)
    return best[1]


def _turn(bits, s):
    return sum(nt.cell(bits, nt.sym_cell(i, s)) << (5 * i) for i in range(16))


def _ref_heur_move(bits, depth, prune=False):
    """The heuristic search by its definition: oriented afterstates, no merge points, a loss at -2**22 (or, with
    prune, macroxue's pass score and loss value beside his tables; prune="big": his big build's rules, pruning only
    when the root scores at least twice the largest tile's score, and an unpruned search again when the best move
    scores below 0)."""
    s = _ref_orient(bits)
    top = max(nt.cell(bits, i) for i in range(16))
    ts = top << top
    pass_, lost = -np.inf, -float(1 << 22)
    if prune:
        pass_ = -np.inf if _ref_heur(_turn(bits, s)) < (2.0 * ts if prune == "big" else 0) else 2.0 * ts
        lost = -float(max(1 << 17, 2 * ts))
    best, best_v = _ref_heur_root(bits, s, depth, pass_, lost)
    if prune == "big" and best >= 0 and best_v < 0:
        best, best_v = _ref_heur_root(bits, s, depth, -np.inf, -float(1 << 22))
    return best


def _ref_heur_root(bits, s, depth, pass_, lost):

    def chance(after, depth):
        v = float(_ref_heur(after))
        if depth == 0 or v < pass_:
            return v
        empties = [i for i in range(16) if nt.cell(after, i) == 0]
        total = 0.0
        for i in empties:
            total += 0.9 * maxnode(nt.with_cell(after, i, 1), depth - 1)
            total += 0.1 * maxnode(nt.with_cell(after, i, 2), depth - 1)
        return total / len(empties)

    def maxnode(b, depth):
        vals = [chance(nb, depth) for nb in (nt.move(b, d)[0] for d in range(4)) if nb != b]
        return max(vals) if vals else lost

    best, best_v = -1, None
    for d in range(4):
        nb = nt.move(bits, d)[0]
        if nb != bits:
            v = chance(_turn(nb, s), depth)
            if best_v is None or v > best_v:
                best, best_v = d, v
    return best, best_v


def _late_boards(rng, count, empties=4, ordered=False):
    """Crowded boards holding a 32768 and a 16384 (or a 65536), small tiles elsewhere, with a legal move.
    ordered: tiles descending from the top left, two small ones swapped (boards the heuristic scores above 0)."""
    out = []
    while len(out) < count:
        g = rng.integers(1, 9, size=16).astype(np.uint8)
        cells = rng.permutation(16)
        g[cells[0]], g[cells[1]] = (16 if len(out) % 3 == 2 else 15), 14
        g[cells[2:2 + empties]] = 0
        if ordered:
            g = np.sort(g)[::-1].copy()
            i, j = rng.choice(np.arange(2, 16 - empties), 2, replace=False)
            g[i], g[j] = g[j], g[i]
            k, turned = int(rng.integers(8)), g.copy()               # in any of the eight orientations
            for c in range(16):
                turned[nt.sym_cell(c, k)] = g[c]
            g = turned
        b = nt.to_bits(g.reshape(4, 4))
        if nt.afterstates_valid(b):
            out.append(b)
    return out


def test_heuristic_score_is_macroxues_formula():
    row = nt.to_bits(np.array([[15, 14, 13, 12], [0] * 4, [0] * 4, [0] * 4], dtype=np.uint8))
    assert nt.heuristic_score(row) == 3457024          # 1703936 for the row, 983040 + 458752 + 212992 + 98304 down
    two = nt.to_bits(np.array([[0, 1, 0, 0], [0] * 4, [0] * 4, [0] * 4], dtype=np.uint8))
    assert nt.heuristic_score(two) == -18               # (0 - 2) x 12 + 2 along the row, 2 + 2 down its column
    rng = np.random.default_rng(31)
    for _ in range(200):
        b = nt.to_bits(rand_exps(rng, density=0.8, max_exp=17))
        assert nt.heuristic_score(b) == _ref_heur(b)


def test_heuristic_orientation_puts_the_largest_tile_at_the_top_left():
    rng = np.random.default_rng(32)
    for b in _late_boards(rng, 40, empties=3):
        s, o = nt.heuristic_orient(b)
        assert s == _ref_orient(b) and o == _turn(b, s)
    corner = nt.to_bits(np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [15, 14, 0, 0]], dtype=np.uint8))
    s, o = nt.heuristic_orient(corner)
    assert nt.cell(o, 0) == 15 and nt.cell(o, 1) == 14     # the 16384 along the first row, not down the column


@pytest.mark.parametrize("prune", [False, True, "big"])
def test_heuristic_late_search_is_the_plain_expectimax_over_the_heuristic(prune):
    n = nt.NTupleNet(patterns=[[0, 1, 2, 3], [4, 5, 6, 7]], tc=False)
    n.heuristic_late, n.heuristic_prune = True, bool(prune)
    if prune == "big":
        nt.lib().nt_set_heur(n._h, 1, 2)               # his big build's rules, orientation turned per move
    rng = np.random.default_rng(33)
    for b in _late_boards(rng, 24) + _late_boards(rng, 24, ordered=True):
        assert n.best_move(b, depth=1) == _ref_heur_move(b, 1, prune)
    for b in _late_boards(rng, 2, empties=3) + _late_boards(rng, 2, empties=3, ordered=True):
        assert n.best_move(b, depth=2) == _ref_heur_move(b, 2, prune)
    n.close()


def test_heuristic_big_rules_differ_from_the_small_ones_where_they_should():
    """His big build's rules (prune mode 2) match the reference on crowded late boards, and on some of them choose
    another move than the small build's rules (the root is not healthy enough to prune, or the search is redone)."""
    n = nt.NTupleNet(patterns=[[0, 1, 2, 3], [4, 5, 6, 7]], tc=False)
    n.heuristic_late = True
    nt.lib().nt_set_heur(n._h, 1, 2)
    differ = 0
    for seed in (41, 43, 44, 50):
        rng = np.random.default_rng(seed)
        for b in _late_boards(rng, 30, empties=2) + _late_boards(rng, 30, empties=2, ordered=True):
            m = n.best_move(b, depth=1)
            assert m == _ref_heur_move(b, 1, "big")
            differ += m != _ref_heur_move(b, 1, True)
    assert differ > 0
    n.close()


def test_heuristic_late_leaves_other_boards_and_the_default_alone():
    """Off by default; on, it changes late boards only."""
    n = _trained_small_net()
    assert not n.heuristic_late and not n.heuristic_prune
    rng = np.random.default_rng(34)
    early = [nt.to_bits(rand_exps(rng, density=0.7, max_exp=14)) for _ in range(30)]
    early = [b for b in early if not ({14, 15} <= {nt.cell(b, i) for i in range(16)})]
    late = _late_boards(rng, 30)
    off = [n.best_move(b, depth=1) for b in early + late]
    n.heuristic_late = True
    on = [n.best_move(b, depth=1) for b in early + late]
    assert on[:len(early)] == off[:len(early)]
    assert on[len(early):] != off[len(early):]
    assert on[len(early):] == [_ref_heur_move(b, 1) for b in late]
    n.heuristic_late = False
    assert [n.best_move(b, depth=1) for b in early + late] == off
    n.close()


def _heuristic_player(how):
    n = nt.NTupleNet(patterns=[[0, 1, 2, 3], [4, 5, 6, 7]], tc=False)       # empty tables: the heuristic alone
    n.heuristic_orientation, n.heuristic_late = how, True
    return n


def test_sticky_heuristic_keeps_its_first_orientation_for_the_whole_game():
    """Sticky: heuristic_orient of the game's first late board, every move after, even where heuristic_orient of
    the board itself has changed; the same search as the board's own orientation on the board turned that way."""
    sticky, own = _heuristic_player("sticky"), _heuristic_player("own")
    assert sticky.heuristic_late and sticky.heuristic_orientation == "sticky" and own.heuristic_orientation == "own"
    rng, seed, flips = np.random.default_rng(36), 0, 0
    for start in _late_boards(rng, 3, empties=6):
        sticky.heuristic_new_game()
        assert sticky.heuristic_game_orientation == -1
        s0, b = nt.heuristic_orient(start)[0], start
        for _ in range(120):
            d = sticky.best_move(b, depth=1)
            if d < 0:
                break
            assert sticky.heuristic_game_orientation == s0
            d_own = own.best_move(_turn(b, s0), depth=1)
            assert _turn(nt.move(b, d)[0], s0) == nt.move(_turn(b, s0), d_own)[0]
            flips += nt.heuristic_orient(b)[0] != s0
            b, seed = nt.spawn(nt.move(b, d)[0], seed), seed + 1
    assert flips > 0                              # boards where turning per move would have picked another corner
    sticky.close()
    own.close()


def test_sticky_heuristic_orientation_is_chosen_afresh_every_game():
    """A game's play does not depend on the games played before it on the same thread."""
    n = _heuristic_player("sticky")
    boards = _late_boards(np.random.default_rng(35), 6, empties=8, ordered=True)
    game = boards[1]                              # game 1 of seed 7 from it: hundreds of moves
    before = [b for b in boards if nt.heuristic_orient(b)[0] != nt.heuristic_orient(game)[0]][:3]
    assert len(before) == 3
    runs = [n.play_boards([b, game], depth=1, threads=1, seed=7) for b in [game] + before]
    runs.append(n.play_boards([before[0], game], depth=1, threads=2, seed=7))
    for r in runs[1:]:
        for k in ("max_tile", "score", "moves", "seen"):
            assert r[k][1] == runs[0][k][1]
    turn = _heuristic_player("turn")
    assert turn.play_boards([game, game], depth=1, threads=1, seed=7)["moves"][1] != runs[0]["moves"][1]
    turn.close()                                  # (sticky plays this game differently from turning per move)
    n.close()


def test_heuristic_orientation_and_depth_change_nothing_else():
    """Off: play is unchanged whatever the orientation and depth say; on, early boards are unchanged and the late
    ones are searched at heuristic_depth."""
    n = _trained_small_net()
    assert n.heuristic_orientation == "turn" and n.heuristic_depth == 0
    plain = n.play(4, depth=1, threads=2, seed=9)["scores"].tolist()
    rng = np.random.default_rng(37)
    early = [nt.to_bits(rand_exps(rng, density=0.7, max_exp=14)) for _ in range(30)]
    early = [b for b in early if not ({14, 15} <= {nt.cell(b, i) for i in range(16)})]
    off = [n.best_move(b, depth=1) for b in early]
    n.heuristic_orientation, n.heuristic_depth = "sticky", 2
    assert not n.heuristic_late and n.heuristic_orientation == "sticky"
    assert n.play(4, depth=1, threads=2, seed=9)["scores"].tolist() == plain
    n.heuristic_late = True                       # this small network's games never reach a late board
    n.heuristic_prune = True
    assert n.heuristic_orientation == "sticky" and n.heuristic_late and n.heuristic_prune
    assert n.play(4, depth=1, threads=2, seed=9)["scores"].tolist() == plain
    assert [n.best_move(b, depth=1) for b in early] == off
    n.heuristic_orientation, n.heuristic_prune = "turn", False
    for b in _late_boards(rng, 2, empties=3):
        assert n.best_move(b, depth=1) == _ref_heur_move(b, 2)
    n.heuristic_late = False
    assert n.heuristic_orientation == "turn" and n.heuristic_depth == 2
    with pytest.raises(ValueError):
        n.heuristic_orientation = "corner"
    n.close()
