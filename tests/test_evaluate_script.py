import json
import sys
from pathlib import Path

import numpy as np

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
