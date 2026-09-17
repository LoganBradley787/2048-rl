import csv
import json

from game2048 import ntuple as nt
from game2048 import ntuple_train as tr


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_train_cli_writes_log_weights_and_state(tmp_path):
    tr.main(["--games", "300", "--chunk", "100", "--threads", "2", "--out", str(tmp_path),
             "--eval-every-min", "0", "--save-every-min", "0"])
    log = rows(tmp_path / "log.csv")
    assert [int(r["games"]) for r in log] == [100, 200, 300]
    assert all(float(r["mean_score"]) > 0 and float(r["moves_per_s"]) > 0 for r in log)
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["games"] == 300 and state["moves"] == sum(int(r["chunk_moves"]) for r in log)
    assert state["threads"] == 2 and state["chunk"] == 100
    net = nt.NTupleNet.load(tmp_path / "latest.bin")
    assert net.patterns == nt.DEFAULT_PATTERNS and net.tc
    net.close()
    small = nt.NTupleNet.load(tmp_path / "weights.bin")  # deployment copy, no learning state
    assert small.tc is False
    assert (tmp_path / "weights.bin").stat().st_size < (tmp_path / "latest.bin").stat().st_size / 2
    small.close()


def test_train_cli_resume_continues_counters_and_log(tmp_path):
    tr.main(["--games", "200", "--chunk", "100", "--threads", "2", "--out", str(tmp_path),
             "--eval-every-min", "0", "--save-every-min", "0"])
    first = rows(tmp_path / "log.csv")
    tr.main(["--games", "200", "--chunk", "100", "--threads", "2", "--out", str(tmp_path),
             "--resume", str(tmp_path / "latest.bin"), "--eval-every-min", "0", "--save-every-min", "0"])
    log = rows(tmp_path / "log.csv")
    assert [int(r["games"]) for r in log] == [100, 200, 300, 400]
    assert log[:2] == first
    assert json.loads((tmp_path / "state.json").read_text())["games"] == 400


def test_train_cli_can_run_an_expectimax_eval(tmp_path):
    tr.main(["--games", "100", "--chunk", "100", "--threads", "2", "--out", str(tmp_path),
             "--eval-every-min", "0", "--save-every-min", "0", "--eval-at-end",
             "--eval-games", "4", "--eval-depth", "1"])
    ev = rows(tmp_path / "eval.csv")
    assert len(ev) == 1 and int(ev[0]["depth"]) == 1 and int(ev[0]["eval_games"]) == 4
    assert float(ev[0]["mean_score"]) > 0


def test_train_cli_stages_flag_builds_a_multistage_network(tmp_path):
    tr.main(["--games", "100", "--chunk", "100", "--threads", "2", "--out", str(tmp_path),
             "--eval-every-min", "0", "--save-every-min", "0", "--stages", "16384,24576"])
    assert json.loads((tmp_path / "state.json").read_text())["boundaries"] == [16384, 24576]
    net = nt.NTupleNet.load(tmp_path / "latest.bin")
    assert net.boundaries == [16384, 24576]
    net.close()


def test_train_cli_lock_memory_flag_is_recorded(tmp_path):
    tr.main(["--games", "100", "--chunk", "100", "--threads", "2", "--out", str(tmp_path),
             "--eval-every-min", "0", "--save-every-min", "0", "--lock-memory"])
    assert json.loads((tmp_path / "state.json").read_text())["locked"] is True


def test_train_cli_patterns_flag(tmp_path):
    tr.main(["--games", "100", "--chunk", "100", "--threads", "2", "--out", str(tmp_path),
             "--eval-every-min", "0", "--save-every-min", "0", "--patterns", "8x6"])
    assert len(json.loads((tmp_path / "state.json").read_text())["patterns"]) == 8


def test_time_budget_ignores_wall_clock_jumps(tmp_path, monkeypatch):
    """A laptop sleeping must not eat the training budget: the budget uses a clock
    that pauses with the machine, not time.time()."""
    import itertools
    fake = itertools.count(1_000_000_000, 36_000)      # every call jumps 10 hours
    monkeypatch.setattr(tr.time, "time", lambda: float(next(fake)))
    tr.main(["--games", "300", "--chunk", "100", "--threads", "2", "--out", str(tmp_path),
             "--hours", "0.05", "--eval-every-min", "0", "--save-every-min", "0"])
    assert json.loads((tmp_path / "state.json").read_text())["games"] == 300


def test_train_cli_choose_flag_trains_and_evaluates_in_choose_mode(tmp_path):
    import numpy as np
    from game2048 import ntuple as nt
    starts = tmp_path / "starts.npy"
    nt.save_starts(starts, np.array([[[10, 9, 8, 7], [3, 4, 5, 6], [2, 1, 0, 0], [0, 0, 0, 0]]], dtype=np.uint8))
    tr.main(["--games", "60", "--chunk", "30", "--threads", "2", "--out", str(tmp_path), "--choose",
             "--max-moves", "500", "--eval-every-min", "0", "--save-every-min", "0", "--eval-at-end",
             "--eval-games", "3", "--eval-depth", "1", "--eval-prefix", "5", "--explore", "0.02",
             "--starts", str(starts), "--start-frac", "0.5"])
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["choose"] is True and state["games"] == 60 and state["eval_prefix"] == 5
    assert state["explore"] == 0.02 and state["starts"] == str(starts) and state["start_frac"] == 0.5
    ev = rows(tmp_path / "eval.csv")
    assert len(ev) == 1 and ev[0]["mode"] == "choose"
