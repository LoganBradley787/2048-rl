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
    locked = json.loads((tmp_path / "state.json").read_text())["locked"]
    assert locked in (True, False)    # recorded either way; False where the OS refuses to pin that much memory


def test_train_cli_patterns_flag(tmp_path):
    tr.main(["--games", "100", "--chunk", "100", "--threads", "2", "--out", str(tmp_path),
             "--eval-every-min", "0", "--save-every-min", "0", "--patterns", "8x6"])
    assert len(json.loads((tmp_path / "state.json").read_text())["patterns"]) == 8


def test_train_cli_v_init_starts_an_optimistic_network_and_remembers_it(tmp_path):
    import numpy as np
    args = ["--chunk", "50", "--threads", "2", "--out", str(tmp_path), "--no-tc", "--alpha-plain", "0.1",
            "--eval-every-min", "0", "--save-every-min", "0"]
    tr.main(["--games", "50", "--v-init", "20000", *args])
    assert json.loads((tmp_path / "state.json").read_text())["v_init"] == 20000
    unseen = nt.to_bits(np.full((4, 4), 15, dtype=np.uint8))
    net = nt.NTupleNet.load(tmp_path / "latest.bin")
    assert abs(net.value(unseen) - 20000) < 1
    net.close()
    tr.main(["--games", "50", "--resume", str(tmp_path / "latest.bin"), *args])
    assert json.loads((tmp_path / "state.json").read_text())["v_init"] == 20000   # carried across a resume


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


def test_kill_stops_a_detached_run_with_a_save(tmp_path):
    """A run started in the background ignores SIGINT; SIGTERM must still save and exit."""
    import signal
    import subprocess
    import sys
    import time
    proc = subprocess.Popen(
        [sys.executable, "-m", "game2048.ntuple_train", "--hours", "1", "--chunk", "20", "--threads", "1",
         "--out", str(tmp_path), "--eval-every-min", "0", "--save-every-min", "0"],
        stdout=subprocess.PIPE, text=True, preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
    assert proc.stdout.readline().startswith("games 20")      # training is under way
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode == 0 and "interrupted; saving" in out
    assert json.loads((tmp_path / "state.json").read_text())["games"] >= 20
    assert (tmp_path / "latest.bin").exists()


def test_train_cli_trains_a_later_stage_from_saved_boards(tmp_path):
    base_dir, out = tmp_path / "base", tmp_path / "stage2"
    common = ["--chunk", "100", "--threads", "2", "--save-every-min", "0", "--no-tc", "--alpha-plain", "0.1"]
    tr.main(["--games", "300", "--out", str(base_dir), "--eval-every-min", "0", *common])
    base = nt.NTupleNet.load(base_dir / "weights.bin")
    grids, _ = base.collect_starts(games=40, tile=128, threads=2, seed=1)
    base.close()
    assert len(grids) > 0
    nt.save_starts(tmp_path / "starts.npy", grids)

    stage = ["--out", str(out), "--eval-every-min", "0", "--eval-at-end", "--eval-games", "4", "--eval-depth", "1", *common]
    tr.main(["--games", "200", "--starts", str(tmp_path / "starts.npy"), "--base", str(base_dir / "weights.bin"),
             "--stage-tile", "128", "--v-init", "2000", *stage])
    log = rows(out / "log.csv")
    assert all(int(r["max_tile"]) >= 128 for r in log)           # every game began with the 128 on the board
    state = json.loads((out / "state.json").read_text())
    assert state["starts"] == str(tmp_path / "starts.npy") and state["stage_tile"] == 128
    assert len(rows(out / "eval.csv")) == 1                      # whole games, played by both networks

    tr.main(["--games", "100", "--resume", str(out / "latest.bin"), *stage])   # the boards and the base carry over
    assert [int(r["games"]) for r in rows(out / "log.csv")] == [100, 200, 300]
    assert all(int(r["max_tile"]) >= 128 for r in rows(out / "log.csv"))
    assert len(rows(out / "eval.csv")) == 2
    assert json.loads((out / "state.json").read_text())["base"] == str(base_dir / "weights.bin")


def test_total_moves_is_a_budget_across_resumes_and_saves_keep_the_previous_checkpoint(tmp_path):
    common = ["--chunk", "50", "--threads", "2", "--out", str(tmp_path), "--eval-every-min", "0", "--no-tc",
              "--save-every-min", "0", "--keep-previous"]
    tr.main(["--games", "100", *common])
    first = json.loads((tmp_path / "state.json").read_text())["moves"]
    assert not (tmp_path / "previous.bin").exists()              # nothing came before the first save
    budget = first * 3
    tr.main(["--total-moves", str(budget), "--resume", str(tmp_path / "latest.bin"), *common])
    state = json.loads((tmp_path / "state.json").read_text())
    last_chunk = int(rows(tmp_path / "log.csv")[-1]["chunk_moves"])
    assert budget <= state["moves"] < budget + last_chunk        # stopped at the first chunk past the budget
    assert (tmp_path / "previous.bin").exists()
    assert not list(tmp_path.glob("*.tmp"))
    nt.NTupleNet.load(tmp_path / "previous.bin").close()         # a complete checkpoint, loadable
    games = state["games"]
    tr.main(["--total-moves", str(budget), "--resume", str(tmp_path / "latest.bin"), *common])   # budget already met
    assert json.loads((tmp_path / "state.json").read_text())["games"] == games
