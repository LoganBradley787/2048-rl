import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import evaluate_ntuple  # noqa: E402

from game2048 import ntuple as nt  # noqa: E402


def test_evaluate_script_covers_cool_mode_tree_and_beam(tmp_path, capsys):
    n = nt.NTupleNet()
    n.train_choose(games=50, threads=2, seed=1, max_moves=300)
    n.save(tmp_path / "w.bin", weights_only=True)
    n.close()
    evaluate_ntuple.main(["--weights", str(tmp_path / "w.bin"), "--choose", "--depths", "1", "--beam", "4:3,4:3:2:2",
                          "--games", "2", "--threads", "2", "--max-moves", "300", "--json", str(tmp_path / "e.json")])
    out = json.loads((tmp_path / "e.json").read_text())
    assert out["meta"]["mode"] == "choose"
    assert set(out["evals"]) == {"tree-1", "beam-4:3", "beam-4:3:2:2"}
    assert all(v["games"] == 2 and v["mean_score"] > 0 for v in out["evals"].values())
    assert "beam-4:3:2:2" in capsys.readouterr().out


def test_evaluate_games_resumes_and_shards_add_up_to_the_same_games(tmp_path, capsys):
    import evaluate_games

    n = nt.NTupleNet(patterns=[[0, 1, 2, 3], [4, 5, 6, 7]], tc=False)
    n.train(games=300, threads=2, seed=1, alpha_plain=0.1)
    n.save(tmp_path / "w.bin", weights_only=True)
    whole = n.play(games=6, depth=1, threads=2, seed=11)
    assert (n.play(games=2, depth=1, threads=1, seed=11, first=3)["scores"] == whole["scores"][3:5]).all()
    n.close()

    common = ["--weights", str(tmp_path / "w.bin"), "--depth", "1", "--seed", "11", "--threads", "2"]
    evaluate_games.main([*common, "--games", "6", "--shard", "0/2", "--out", str(tmp_path / "a.csv")])
    evaluate_games.main([*common, "--games", "4", "--shard", "1/2", "--out", str(tmp_path / "b.csv")])   # stopped early
    assert sorted(evaluate_games.read_games(tmp_path / "b.csv")) == [1, 3]
    evaluate_games.main([*common, "--games", "6", "--shard", "1/2", "--out", str(tmp_path / "b.csv")])   # resumed
    assert sorted(evaluate_games.read_games(tmp_path / "b.csv")) == [1, 3, 5]

    stats = evaluate_games.summarize([tmp_path / "a.csv", tmp_path / "b.csv"])
    assert stats["games"] == 6 and stats["mean_score"] == whole["scores"].mean()
    evaluate_games.main(["--summary", str(tmp_path / "a.csv"), str(tmp_path / "b.csv")])
    assert "6 games" in capsys.readouterr().out

    with pytest.raises(SystemExit):                              # other settings do not mix into a file
        evaluate_games.main([*common[:2], "--depth", "2", "--seed", "11", "--games", "6", "--out", str(tmp_path / "a.csv")])


def test_evaluate_games_heuristic_and_block_flags(tmp_path):
    """--heuristic, --heuristic-prune, --heuristic-depth and --block-dir play and are part of the settings
    (in a process of their own: block tables load once per process)."""
    import os
    import subprocess

    import evaluate_games
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")])}
    script = [sys.executable, str(root / "examples" / "evaluate_games.py")]
    usage = subprocess.run([*script, "--help"], capture_output=True, text=True, env=env, check=True).stdout
    for flag in ("--heuristic {off,turn,own,sticky}", "--heuristic-prune", "--heuristic-depth", "--block-dir"):
        assert flag in usage
    subprocess.run([sys.executable, str(root / "examples" / "build_endgame.py"), "--block", "b6", "--anchors", "4,5",
                    "--threads", "2", "--out", str(tmp_path / "b6")], check=True, capture_output=True, env=env)
    n = nt.NTupleNet(patterns=[[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 4, 5]], tc=False)
    n.train(200, threads=1, seed=1, alpha_plain=0.1)
    n.save(tmp_path / "w.bin", weights_only=True)
    n.close()
    common = [*script, "--weights", str(tmp_path / "w.bin"), "--games", "2", "--depth", "1", "--threads", "2",
              "--out", str(tmp_path / "e.csv")]
    flags = ["--heuristic", "sticky", "--heuristic-prune", "--heuristic-depth", "2", "--block-dir", str(tmp_path / "b6")]
    r = subprocess.run([*common, *flags], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "2 games" in r.stdout and "block tier:" in r.stdout and "block tier: 0 moves" not in r.stdout
    s = json.loads((tmp_path / "e.json").read_text())
    assert (s["heuristic"], s["heuristic_prune"], s["heuristic_depth"]) == ("sticky", True, 2)
    assert s["block_tables"].startswith("b6 4,5 ")
    r = subprocess.run([*common, *flags[:2]], capture_output=True, text=True, env=env)   # other settings, same file
    assert r.returncode != 0 and "other settings" in r.stderr
    r = subprocess.run([*common[:-1], str(tmp_path / "f.csv"), "--heuristic-prune"], capture_output=True, text=True,
                       env=env)
    assert r.returncode != 0 and "need --heuristic" in r.stderr
    scores = {}                                   # tables from 32768 + 16384 on: this small network's games unchanged
    for name, extra in (("plain", []), ("late", ["--block-dir", str(tmp_path / "b6"), "--block-from", "49152"])):
        r = subprocess.run([*common[:-1], str(tmp_path / f"{name}.csv"), *extra], capture_output=True, text=True,
                           env=env)
        assert r.returncode == 0, r.stderr
        scores[name] = [g["score"] for _, g in sorted(evaluate_games.read_games(tmp_path / f"{name}.csv").items())]
    assert "block tier: 0 moves" in r.stdout and scores["late"] == scores["plain"]
    assert json.loads((tmp_path / "late.json").read_text())["block_from"] == 49152


def test_average_weights_is_the_mean_of_the_files(tmp_path):
    import average_weights

    pats = [[0, 1, 2, 3], [4, 5, 6]]
    boards = [nt.spawn(nt.spawn(0, s), s + 1) for s in range(6)]
    values = []
    for i, v in enumerate((100.0, 400.0, 1000.0)):
        n = nt.NTupleNet(patterns=pats, tc=False, v_init=v)
        n.train(games=30, threads=1, seed=i, alpha_plain=0.1)
        n.save(tmp_path / f"w{i}.bin", weights_only=True)
        values.append([n.value(b) for b in boards])
        if i == 0:
            n.save(tmp_path / "state.bin")
            nt.NTupleNet(patterns=[[0, 1, 2, 3]], tc=False).save(tmp_path / "other.bin", weights_only=True)
        n.close()
    files = [str(tmp_path / f"w{i}.bin") for i in range(3)]
    average_weights.main(["--weights", *files, "--out", str(tmp_path / "avg" / "weights.bin")])
    avg = nt.NTupleNet.load(tmp_path / "avg" / "weights.bin")
    assert avg.patterns == pats and avg.tc is False
    assert [avg.value(b) for b in boards] == pytest.approx(np.mean(values, axis=0), rel=1e-5)
    avg.close()
    with pytest.raises(SystemExit):                              # a different shape
        average_weights.main(["--weights", files[0], str(tmp_path / "other.bin"), "--out", str(tmp_path / "x.bin")])
    n = nt.NTupleNet(patterns=pats)                              # learning state is not something to average
    n.save(tmp_path / "tc.bin")
    n.close()
    with pytest.raises(SystemExit):
        average_weights.main(["--weights", str(tmp_path / "tc.bin"), files[0], "--out", str(tmp_path / "x.bin")])
