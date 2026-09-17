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


def test_stage_is_number_of_mass_boundaries_reached():
    bounds = [16384, 24576]
    assert nt.stage_of(nt.to_bits(board_with([8192, 4096, 2048])), bounds) == 0
    assert nt.stage_of(nt.to_bits(board_with([16384])), bounds) == 1
    assert nt.stage_of(nt.to_bits(board_with([16384, 8192])), bounds) == 2
    assert nt.stage_of(nt.to_bits(board_with([16384, 4096, 2048, 2048])), bounds) == 2
    assert nt.stage_of(nt.to_bits(board_with([2])), []) == 0


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
    assert n.lock_memory() is True
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
            assert len(p) == 6 and len(set(p)) == 6
            orbit = frozenset(frozenset(nt.sym_cell(c, k) for c in p) for k in range(8))
            assert orbit not in orbits, f"{name}: {p} repeats another pattern's symmetry orbit"
            orbits.add(orbit)
    assert nt.PATTERN_SETS["4x6"] == nt.DEFAULT_PATTERNS and len(nt.PATTERN_SETS["8x6"]) == 8


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


def test_play_choose_scores_far_above_random_spawns(net):
    n = nt.NTupleNet()
    n.train(games=2000, threads=4, seed=6)
    normal = n.play(games=8, depth=1, threads=4, seed=3)["scores"].mean()
    stats = n.play_choose(games=8, depth=1, topk=4, threads=4, seed=3, max_moves=3000)
    assert stats["scores"].shape == (8,) and (stats["moves"] > 0).all()
    assert stats["scores"].mean() > 1.5 * normal
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
    n = nt.NTupleNet()
    before = n.play_choose(games=6, depth=1, topk=4, threads=3, seed=9, max_moves=30_000)["scores"].mean()
    stats = n.train_choose(games=1500, threads=4, seed=1, max_moves=100_000)
    assert stats["scores"].shape == (1500,) and (stats["moves"] > 0).all()
    after = n.play_choose(games=6, depth=1, topk=4, threads=3, seed=9, max_moves=30_000)["scores"].mean()
    assert after > 1.5 * before, (before, after)
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
    game.place  # a human-made deviation: the plan no longer applies
    game2 = Game(seed=5, spawn_mode="choose")
    direction, (row, col, value) = agent.choose(game2)
    assert direction in game2.legal_moves()
    assert searches == 4


def test_tiles_to_bits_never_mistakes_small_tiles_for_exponents():
    board = [[2, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 8]]
    assert nt.tiles_to_bits(board) == nt.with_cell(nt.with_cell(nt.with_cell(0, 0, 1), 1, 2), 15, 3)
    assert nt.tiles_to_bits([[0] * 4] * 4) == 0


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
    nt.set_snake_decay(0.5)                                        # the decay is process-wide
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
    for _ in range(5):
        direction, (row, col, value) = agent.choose(game)
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
    try:
        nt.set_snake_decay(0.9)
        assert nt.snake_score(chain) == pytest.approx(sum(t * 0.9 ** k for k, t in enumerate(tiles)))
        nt.set_snake_decay(1.0)
        assert nt.snake_score(chain) == pytest.approx(sum(tiles))
    finally:
        nt.set_snake_decay(0.5)
    assert nt.snake_score(chain) == pytest.approx(sum(t * 0.5 ** k for k, t in enumerate(tiles)))
