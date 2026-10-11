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
