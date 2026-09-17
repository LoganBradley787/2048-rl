import pytest

from game2048.agents import AGENTS, GreedyAgent, RandomAgent, make_agent
from game2048.core import Direction, Game


def board_from(rows):
    g = Game(seed=0)
    g.board = [list(r) for r in rows]
    g.score = 0
    return g


def test_random_agent_only_picks_legal_moves():
    g = board_from([[2, 4, 8, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    agent = RandomAgent(seed=1)
    for _ in range(20):
        assert agent.act(g) == Direction.DOWN


def test_random_agent_is_deterministic_under_seed():
    g = Game(seed=5)
    a, b = RandomAgent(seed=1), RandomAgent(seed=1)
    assert [a.act(g) for _ in range(10)] == [b.act(g) for _ in range(10)]


def test_greedy_agent_picks_highest_immediate_reward():
    # LEFT/RIGHT merge 8+8=16; UP/DOWN merge 2+2=4 in column 0
    g = board_from([[2, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 8, 8]])
    assert GreedyAgent().act(g) in (Direction.LEFT, Direction.RIGHT)


def test_greedy_agent_never_returns_illegal_move():
    g = board_from([[2, 4, 8, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert GreedyAgent().act(g) == Direction.DOWN


def test_registry_builds_agents_by_name():
    assert set(AGENTS) >= {"random", "greedy"}
    assert isinstance(make_agent("random"), RandomAgent)
    assert isinstance(make_agent("greedy"), GreedyAgent)


def test_registry_includes_nn_agent():
    assert "nn" in AGENTS


def test_nn_agent_without_checkpoint_is_file_not_found(monkeypatch, tmp_path):
    monkeypatch.setenv("GAME2048_CHECKPOINT", str(tmp_path / "missing.pt"))
    with pytest.raises(FileNotFoundError):
        make_agent("nn")


def test_registry_includes_ntuple_agent():
    assert "ntuple" in AGENTS


def test_ntuple_agent_without_weights_is_file_not_found(monkeypatch, tmp_path):
    monkeypatch.setenv("GAME2048_NTUPLE", str(tmp_path / "missing.bin"))
    with pytest.raises(FileNotFoundError):
        make_agent("ntuple")


def test_builtin_agents_can_play_a_full_game():
    for name in ("random", "greedy"):
        g = Game(seed=11)
        agent = make_agent(name)
        steps = 0
        while not g.is_over():
            assert g.move(agent.act(g)).moved
            steps += 1
            assert steps < 10_000


def test_registry_includes_the_cool_mode_agent(monkeypatch, tmp_path):
    assert "ntuple-cool" in AGENTS
    monkeypatch.setenv("GAME2048_NTUPLE_COOL", str(tmp_path / "missing.bin"))
    with pytest.raises(FileNotFoundError):
        make_agent("ntuple-cool")


def test_cool_agent_reads_search_settings_saved_beside_its_weights(monkeypatch, tmp_path):
    import json
    from game2048 import ntuple as nt
    n = nt.NTupleNet()
    n.save(tmp_path / "weights.bin", weights_only=True)
    n.close()
    (tmp_path / "best.json").write_text(json.dumps({"width": 8, "depth": 4, "stride": 2, "snake": 0.25, "score": 1}))
    monkeypatch.setenv("GAME2048_NTUPLE_COOL", str(tmp_path / "weights.bin"))
    agent = make_agent("ntuple-cool")
    assert (agent.beam_width, agent.beam_depth, agent.beam_stride, agent.beam_snake) == (8, 4, 2, 0.25)
    (tmp_path / "best.json").unlink()
    agent = make_agent("ntuple-cool")                       # no settings file: the defaults
    assert agent.beam_width == 32 and agent.beam_depth == 12 and agent.beam_stride == 4


def test_snake_agent_needs_no_weights_and_plays_cool_mode():
    from game2048.core import Game
    agent = make_agent("snake")                              # beam search on the snake heuristic alone
    assert agent.beam_width > 0 and agent.beam_snake > 0 and agent.net is not None
    assert agent.beam_snake_decay == 0.9                      # every tile on the path counts, not just the head
    assert agent.beam_tiles == 12                             # 2s first: a 4 only when a 2 dead-ends
    game = Game(seed=2, spawn_mode="choose")
    for _ in range(6):
        direction, (row, col, value) = agent.choose(game)
        assert direction in game.legal_moves() and value in (2, 4)
        game.move(direction)
        game.place(row, col, value)
    assert game.score > 0
    assert "snake" in AGENTS


def test_snake_agent_plays_random_spawns_on_the_snake_heuristic():
    from game2048.core import Game
    agent = make_agent("snake")
    assert agent.net.leaf_snake == 16.0 and agent.depth == 3
    game = Game(seed=3)
    for _ in range(40):
        if game.is_over():
            break
        direction = agent.act(game)
        assert direction in game.legal_moves()
        game.move(direction)
    assert game.score > 0
