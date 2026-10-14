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


def _ref_endgame(g, k, memo):
    """Plain Python version of the engine's endgame rule, for the few-cell tables."""
    n = 16 - k
    key = (bytes(g), k)
    if key in memo:
        return memo[key]
    best = 0.0
    for d in range(4):
        moved, _ = nt.move(nt.to_bits(g.reshape(4, 4)), d)
        a = nt.from_bits(moved).ravel().copy()
        if (a == g).all() or any(a[SNAKE[i]] != 15 - i for i in range(k - 1)):
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
                total += p * _ref_endgame(a.copy(), k, memo)
                a[c] = 0
        best = max(best, total / len(empty) if empty else 0.0)
    if k < 15 and g[SNAKE[k]] == n - 1 and all(g[SNAKE[i]] <= n - 1 for i in range(k + 1, 16)):
        best = max(best, _ref_endgame(g, k + 1, memo))
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
