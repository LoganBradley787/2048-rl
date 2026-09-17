import csv
import json

from game2048 import report


ROWS = [
    {"iteration": 2500, "env_steps": 1_280_000, "games": 2000, "elapsed_s": 50.0, "loss": 0.001,
     "train_mean_score": 4000, "train_max_tile": 1024, "eval_mean_score": 8469.0,
     "eval_median_score": 7064.0, "eval_max_score": 16096, "eval_mean_length": 500.0,
     "eval_max_tile": 1024, "rate_1024": 0.44, "rate_2048": 0.0, "rate_4096": 0.0, "rate_8192": 0.0},
    {"iteration": 5000, "env_steps": 2_560_000, "games": 5000, "elapsed_s": 100.0, "loss": 0.0004,
     "train_mean_score": 11000, "train_max_tile": 2048, "eval_mean_score": 11050.0,
     "eval_median_score": 11626.0, "eval_max_score": 25980, "eval_mean_length": 700.0,
     "eval_max_tile": 2048, "rate_1024": 0.63, "rate_2048": 0.10, "rate_4096": 0.0, "rate_8192": 0.0},
]
FINAL = {"games": 500, "mean_score": 20000.0, "median_score": 19000.0, "max_score": 40000,
         "mean_length": 1000.0, "max_tile": 4096, "tile_counts": {512: 20, 1024: 80, 2048: 350, 4096: 50},
         "rate_1024": 0.96, "rate_2048": 0.8, "rate_4096": 0.1, "rate_8192": 0.0}
CONFIG = {"envs": 512, "batch": 1024, "lr": 0.0003, "width": 128, "hidden": 512}


def test_build_report_is_a_page_with_the_data_embedded():
    html = report.build_report(ROWS, FINAL, CONFIG, elapsed_s=3600.0)
    assert html.startswith("<title>")
    assert "<style>" in html and "<script>" in html
    data = json.loads(html.split("<script id=\"data\" type=\"application/json\">")[1].split("</script>")[0])
    assert data["final"]["mean_score"] == 20000.0
    assert data["log"][1]["rate_2048"] == 0.10
    assert data["config"]["width"] == 128
    assert "80%" in html  # the headline 2048 rate is rendered at rest, not only by script


def test_report_can_show_the_no_symmetry_comparison():
    html = report.build_report(ROWS, FINAL, CONFIG, plain={"mean_score": 23392, "rate_2048": 0.572})
    assert "57%" in html and "23,392" in html
    assert "symmetries" in report.build_report(ROWS, FINAL, CONFIG)  # method text mentions them regardless


def test_main_reads_csv_and_json_and_writes_html(tmp_path):
    log = tmp_path / "log.csv"
    with open(log, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ROWS[0]))
        w.writeheader()
        w.writerows(ROWS)
    ev = tmp_path / "eval.json"
    ev.write_text(json.dumps({**FINAL, "config": CONFIG}))
    out = tmp_path / "report.html"
    report.main(["--log", str(log), "--eval", str(ev), "--out", str(out)])
    html = out.read_text()
    assert "<title>" in html and "20,000" in html


NT_LOG = [
    {"games": 20000, "chunk_moves": 40_000_000, "moves": 40_000_000, "elapsed_s": 6.0, "moves_per_s": 6_000_000,
     "mean_score": 70727, "median_score": 65000, "max_score": 238652, "mean_length": 2000, "max_tile": 16384,
     "rate_2048": 0.94, "rate_4096": 0.70, "rate_8192": 0.15, "rate_16384": 0.0, "rate_32768": 0.0},
    {"games": 40000, "chunk_moves": 80_000_000, "moves": 120_000_000, "elapsed_s": 20.0, "moves_per_s": 6_100_000,
     "mean_score": 105521, "median_score": 100000, "max_score": 303620, "mean_length": 3000, "max_tile": 16384,
     "rate_2048": 0.97, "rate_4096": 0.87, "rate_8192": 0.47, "rate_16384": 0.01, "rate_32768": 0.0},
]
NT_EVALS = {
    "0": {"games": 100, "mean_score": 180000.0, "median_score": 170000.0, "max_score": 400000, "mean_length": 8000.0,
          "max_tile": 16384, "tile_counts": {4096: 5, 8192: 40, 16384: 55}, "rate_1024": 1.0, "rate_2048": 1.0,
          "rate_4096": 1.0, "rate_8192": 0.95, "rate_16384": 0.55, "rate_32768": 0.0},
    "3": {"games": 100, "mean_score": 412345.0, "median_score": 420000.0, "max_score": 700000, "mean_length": 15000.0,
          "max_tile": 32768, "tile_counts": {8192: 5, 16384: 60, 32768: 35}, "rate_1024": 1.0, "rate_2048": 1.0,
          "rate_4096": 1.0, "rate_8192": 1.0, "rate_16384": 0.95, "rate_32768": 0.35},
}
NT_META = {"games": 3_000_000, "moves": 30_000_000_000, "elapsed_s": 7200.0, "threads": 12, "moves_per_s": 6_400_000,
           "patterns": [[0, 1, 2, 3, 4, 5], [4, 5, 6, 7, 8, 9], [0, 1, 2, 4, 5, 6], [4, 5, 6, 8, 9, 10]],
           "tc": True, "boundaries": []}


def test_build_ntuple_report_embeds_data_and_headline():
    html = report.build_ntuple_report(NT_LOG, NT_EVALS, NT_META)
    assert html.startswith("<title>2048 N-Tuple Player</title>")
    data = json.loads(html.split("<script id=\"data\" type=\"application/json\">")[1].split("</script>")[0])
    assert data["evals"]["3"]["rate_32768"] == 0.35
    assert data["log"][1]["rate_8192"] == 0.47
    assert "412,345" in html and "35%" in html          # headline mean and 32768 rate at rest
    assert 'data-v="32768"' in html                      # best tile chip
    assert "3,000,000" in html                          # training games in the meta line


def test_ntuple_report_main_reads_files_and_writes_html(tmp_path):
    log = tmp_path / "log.csv"
    with open(log, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(NT_LOG[0]))
        w.writeheader()
        w.writerows(NT_LOG)
    (tmp_path / "evals.json").write_text(json.dumps({"evals": NT_EVALS, "meta": NT_META}))
    out = tmp_path / "ntuple.html"
    report.ntuple_main(["--log", str(log), "--evals", str(tmp_path / "evals.json"), "--out", str(out)])
    assert "<title>2048 N-Tuple Player</title>" in out.read_text()
