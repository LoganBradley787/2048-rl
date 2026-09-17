"""Build a self-contained HTML report from a training log and a final evaluation.

    uv run python examples/evaluate.py --games 500 --symmetric --json checkpoints/eval.json
    uv run python -m game2048.report --log checkpoints/log.csv --eval checkpoints/eval.json
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

NUMERIC = {
    "iteration", "env_steps", "games", "elapsed_s", "loss", "train_mean_score", "train_max_tile",
    "eval_mean_score", "eval_median_score", "eval_max_score", "eval_mean_length", "eval_max_tile",
    "rate_1024", "rate_2048", "rate_4096", "rate_8192",
}


def _num(v):
    if v is None or v == "":
        return None
    return float(v)


def read_log(path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [{k: (_num(v) if k in NUMERIC else v) for k, v in r.items()} for r in rows]


def fmt_int(n) -> str:
    return f"{int(round(n)):,}"


def fmt_pct(x) -> str:
    return f"{x * 100:.0f}%" if x >= 0.1 or x == 0 else f"{x * 100:.1f}%"


def fmt_duration(s) -> str:
    s = int(s)
    h, m = divmod(s // 60, 60)
    return f"{h}h {m:02d}m" if h else f"{m} min"


def tile(v) -> str:
    v = int(v)
    return f'<span class="tile" data-v="{v}">{v}</span>'


def build_report(rows: list[dict], final: dict, config: dict, elapsed_s: float | None = None,
                 trained_on: str | None = None, best_env_steps: float | None = None,
                 plain: dict | None = None) -> str:
    rows = [{k: (_num(v) if k in NUMERIC and not isinstance(v, (int, float)) else v)
             for k, v in r.items()} for r in rows]
    last = rows[-1] if rows else {}
    elapsed = elapsed_s if elapsed_s is not None else (last.get("elapsed_s") or 0)
    env_steps = last.get("env_steps") or 0
    games = last.get("games") or 0
    counts = {int(k): int(v) for k, v in final.get("tile_counts", {}).items()}
    n_games = final.get("games") or sum(counts.values()) or 1
    best_tile = max(counts) if counts else int(final.get("max_tile", 0))

    curve_rows = "".join(
        f"<tr><td>{r['env_steps'] / 1e6:.1f}M</td><td>{fmt_int(r['eval_mean_score'])}</td>"
        f"<td>{fmt_int(r['eval_median_score'])}</td><td>{fmt_int(r['eval_max_score'])}</td>"
        f"<td>{fmt_pct(r['rate_1024'])}</td><td>{fmt_pct(r['rate_2048'])}</td>"
        f"<td>{fmt_pct(r['rate_4096'])}</td></tr>"
        for r in rows
    )
    hist_rows = "".join(
        f"<tr><td>{tile(t)}</td><td>{c}</td><td>{fmt_pct(c / n_games)}</td></tr>"
        for t, c in sorted(counts.items())
    )
    cfg_items = [
        ("Parallel boards", config.get("envs")), ("Batch size", config.get("batch")),
        ("Gradient steps per iteration", config.get("grad_steps")), ("Learning rate", config.get("lr")),
        ("Conv channels", config.get("width")), ("Hidden units", config.get("hidden")),
        ("Replay buffer", fmt_int(config["buffer"]) if config.get("buffer") else None),
        ("Target Polyak tau", config.get("tau")),
        ("Reward scale", f"1/{round(1 / config['reward_scale'])}" if config.get("reward_scale") else None),
        ("Device", trained_on),
    ]
    cfg_html = "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in cfg_items if v is not None)

    data = json.dumps({"log": rows, "final": final, "config": config})
    page = TEMPLATE
    for key, val in {
        "DATA": data,
        "DATE": date.today().strftime("%B %-d, %Y"),
        "DURATION": fmt_duration(elapsed),
        "MOVES_M": f"{env_steps / 1e6:.1f}M",
        "GAMES": fmt_int(games),
        "BEST_AT": (f"<span>best checkpoint after {best_env_steps / 1e6:.1f}M moves</span>"
                    if best_env_steps else ""),
        "RATE_2048": fmt_pct(final.get("rate_2048", 0)),
        "PLAIN_NOTE": (f'<p class="note">Same weights without averaging the eight board symmetries: '
                       f'{fmt_pct(plain["rate_2048"])} reach 2048, mean score {fmt_int(plain["mean_score"])}.</p>'
                       if plain else ""),
        "N_GAMES": fmt_int(n_games),
        "MEAN": fmt_int(final.get("mean_score", 0)),
        "MEDIAN": fmt_int(final.get("median_score", 0)),
        "MAX": fmt_int(final.get("max_score", 0)),
        "LENGTH": fmt_int(final.get("mean_length", 0)),
        "BEST_TILE": tile(best_tile) if best_tile else "",
        "RATE_4096": fmt_pct(final.get("rate_4096", 0)),
        "RATE_1024": fmt_pct(final.get("rate_1024", 0)),
        "CURVE_ROWS": curve_rows,
        "HIST_ROWS": hist_rows,
        "CONFIG": cfg_html,
    }.items():
        page = page.replace("{{" + key + "}}", str(val))
    return page


TEMPLATE = r"""<title>2048 Afterstate Net</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Chivo:wght@400;700;900&family=Source+Sans+3:wght@400;600&display=swap">
<style>
:root {
  color-scheme: light;
  --ground: #fbf8f1; --surface: #fffdf8; --ink: #1f1b16; --ink-2: #5c554b; --muted: #8a8276;
  --grid: #ebe5d9; --baseline: #cfc7b8; --ring: rgba(31,27,22,.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --accent: #edc22e;
  --tip-bg: #1f1b16; --tip-ink: #fbf8f1;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --ground: #1c1a15; --surface: #242119; --ink: #f3eee4; --ink-2: #c9c1b2; --muted: #8f887b;
    --grid: #33302a; --baseline: #46423a; --ring: rgba(243,238,228,.12);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70;
    --tip-bg: #f3eee4; --tip-ink: #1c1a15;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --ground: #1c1a15; --surface: #242119; --ink: #f3eee4; --ink-2: #c9c1b2; --muted: #8f887b;
  --grid: #33302a; --baseline: #46423a; --ring: rgba(243,238,228,.12);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70;
  --tip-bg: #f3eee4; --tip-ink: #1c1a15;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--ground); color: var(--ink);
  font-family: "Source Sans 3", "Helvetica Neue", Arial, sans-serif; font-size: 17px; line-height: 1.5;
  padding-block: 40px 72px; padding-inline: 20px;
}
h1, h2, .figure, .kpi .value { font-family: "Chivo", "Helvetica Neue", Arial, sans-serif; }
h1 { font-size: 2rem; font-weight: 900; letter-spacing: -.01em; margin: 0; text-wrap: balance; line-height: 1.1; }
h2 { font-size: 1.15rem; font-weight: 700; margin: 0 0 4px; }
p { max-width: 66ch; }
.wrap { max-width: 920px; margin: 0 auto; display: flex; flex-direction: column; gap: 40px; }
.eyebrow { font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); font-weight: 600; }
header .meta { color: var(--ink-2); margin-top: 8px; }
header .meta span + span::before { content: "·"; margin: 0 .5em; color: var(--muted); }

.hero { display: flex; align-items: center; gap: 28px; flex-wrap: wrap; }
.figure { font-size: clamp(64px, 12vw, 112px); font-weight: 900; line-height: .95; letter-spacing: -.03em; }
.hero .lede { max-width: 30ch; color: var(--ink-2); font-size: 1.1rem; margin: 0; }
.hero .lede strong { color: var(--ink); }
.note { margin: -24px 0 0; color: var(--ink-2); font-size: .95rem; }

.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.kpi { background: var(--surface); border: 1px solid var(--ring); border-radius: 6px; padding: 14px 16px; }
.kpi .label { font-size: .8rem; color: var(--muted); font-weight: 600; letter-spacing: .02em; }
.kpi .value { font-size: 1.7rem; font-weight: 700; line-height: 1.2; margin-top: 4px; }
.kpi .sub { font-size: .85rem; color: var(--ink-2); }

.tile {
  display: inline-flex; align-items: center; justify-content: center; vertical-align: middle;
  min-width: 2.6em; height: 2.6em; padding: 0 .3em; border-radius: 4px; font-family: "Chivo", sans-serif;
  font-weight: 900; font-size: .8em; color: #776e65; background: #eee4da;
}
.tile[data-v="4"] { background: #ede0c8; }
.tile[data-v="8"] { background: #f2b179; color: #f9f6f2; }
.tile[data-v="16"] { background: #f59563; color: #f9f6f2; }
.tile[data-v="32"] { background: #f67c5f; color: #f9f6f2; }
.tile[data-v="64"] { background: #f65e3b; color: #f9f6f2; }
.tile[data-v="128"] { background: #edcf72; color: #f9f6f2; }
.tile[data-v="256"] { background: #edcc61; color: #f9f6f2; }
.tile[data-v="512"] { background: #edc850; color: #f9f6f2; }
.tile[data-v="1024"] { background: #edc53f; color: #f9f6f2; }
.tile[data-v="2048"] { background: #edc22e; color: #f9f6f2; }
.tile[data-v="4096"], .tile[data-v="8192"], .tile[data-v="16384"], .tile[data-v="32768"], .tile[data-v="65536"] { background: #3c3a32; color: #f9f6f2; }
.hero .tile { font-size: 2.2rem; box-shadow: 0 0 34px 6px rgba(237,194,46,.35); }

section.chart { display: flex; flex-direction: column; gap: 10px; }
.chart-head { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; flex-wrap: wrap; }
.chart-head p { margin: 0; color: var(--ink-2); font-size: .95rem; }
.legend { display: flex; gap: 18px; font-size: .85rem; color: var(--ink-2); flex-wrap: wrap; }
.legend span::before { content: ""; display: inline-block; width: 18px; height: 2px; vertical-align: middle; margin-right: 6px; border-radius: 1px; }
.legend .k1::before { background: var(--s1); } .legend .k2::before { background: var(--s2); } .legend .k3::before { background: var(--s3); }
.plot { position: relative; background: var(--surface); border: 1px solid var(--ring); border-radius: 6px; padding: 8px 4px 2px; }
.plot svg { width: 100%; height: auto; display: block; font-family: "Source Sans 3", sans-serif; }
.plot .overlay:focus { outline: none; }
.plot .overlay:focus-visible + .focus-ring { stroke: var(--s1); }
svg text { fill: var(--ink-2); font-size: 12px; font-variant-numeric: tabular-nums; }
svg .grid { stroke: var(--grid); stroke-width: 1; }
svg .baseline { stroke: var(--baseline); stroke-width: 1; }
svg .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
svg .l1 { stroke: var(--s1); } svg .l2 { stroke: var(--s2); } svg .l3 { stroke: var(--s3); }
svg .area { fill: var(--s1); opacity: .1; }
svg .dot { stroke: var(--surface); stroke-width: 2; }
svg .d1 { fill: var(--s1); } svg .d2 { fill: var(--s2); } svg .d3 { fill: var(--s3); }
svg .end { fill: var(--ink); font-weight: 600; }
svg .cross { stroke: var(--muted); stroke-width: 1; opacity: 0; }
svg .bar { fill: var(--s1); }
svg .bar.hot { opacity: .8; }
svg .cap { fill: var(--ink); font-weight: 600; text-anchor: middle; }
svg .xt { text-anchor: middle; } svg .yt { text-anchor: end; }
svg .axis-title { fill: var(--muted); font-size: 11px; letter-spacing: .04em; text-transform: uppercase; }
svg .tilebox { rx: 3; }
svg .tilelbl { font-family: "Chivo", sans-serif; font-weight: 900; text-anchor: middle; font-size: 11px; }
.tip {
  position: absolute; pointer-events: none; background: var(--tip-bg); color: var(--tip-ink);
  border-radius: 5px; padding: 8px 10px; font-size: .85rem; line-height: 1.35; opacity: 0; transition: opacity .08s;
  min-width: 140px; z-index: 2;
}
.tip.on { opacity: 1; }
.tip .x { color: inherit; opacity: .7; font-size: .78rem; margin-bottom: 2px; }
.tip .row { display: flex; justify-content: space-between; gap: 14px; }
.tip .row b { font-variant-numeric: tabular-nums; }
.tip .row .k { display: inline-block; width: 12px; height: 2px; vertical-align: middle; margin-right: 6px; }
details { font-size: .9rem; color: var(--ink-2); }
summary { cursor: pointer; font-weight: 600; }
table { border-collapse: collapse; margin-top: 8px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 4px 12px 4px 0; border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-size: .8rem; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }
.table-wrap { overflow-x: auto; }
.method p { margin: 0 0 12px; }
dl { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px 20px; margin: 16px 0 0; }
dl div { display: flex; flex-direction: column; gap: 2px; padding: 8px 0; border-top: 1px solid var(--grid); }
dt { font-size: .8rem; color: var(--muted); font-weight: 600; }
dd { margin: 0; font-weight: 600; font-variant-numeric: tabular-nums; }
@media (prefers-reduced-motion: reduce) { .tip { transition: none; } }
@media (max-width: 480px) { body { font-size: 16px; } .hero { gap: 18px; } }
</style>
<div class="wrap">
  <header>
    <div class="eyebrow">Training report</div>
    <h1>A value network that plays 2048</h1>
    <div class="meta"><span>{{DATE}}</span><span>{{DURATION}} of training</span><span>{{MOVES_M}} moves</span><span>{{GAMES}} games played</span>{{BEST_AT}}</div>
  </header>

  <section class="hero">
    <div class="figure">{{RATE_2048}}</div>
    {{BEST_TILE}}
    <p class="lede">of <strong>{{N_GAMES}} evaluation games</strong> reached the 2048 tile, playing greedily from a single learned value function with no search.</p>
  </section>
  {{PLAIN_NOTE}}

  <section class="kpis">
    <div class="kpi"><div class="label">Mean score</div><div class="value">{{MEAN}}</div><div class="sub">median {{MEDIAN}}</div></div>
    <div class="kpi"><div class="label">Best game</div><div class="value">{{MAX}}</div><div class="sub">points</div></div>
    <div class="kpi"><div class="label">Reached 1024</div><div class="value">{{RATE_1024}}</div><div class="sub">of games</div></div>
    <div class="kpi"><div class="label">Reached 4096</div><div class="value">{{RATE_4096}}</div><div class="sub">of games</div></div>
    <div class="kpi"><div class="label">Average game</div><div class="value">{{LENGTH}}</div><div class="sub">moves</div></div>
  </section>

  <section class="chart">
    <div class="chart-head"><div><h2>Learning curve</h2><p>Mean score over 200 fixed-seed evaluation games, measured as training progressed.</p></div></div>
    <div class="plot" id="curve"></div>
    <details><summary>Table view</summary><div class="table-wrap"><table>
      <thead><tr><th>Moves</th><th>Mean</th><th>Median</th><th>Best</th><th>1024</th><th>2048</th><th>4096</th></tr></thead>
      <tbody>{{CURVE_ROWS}}</tbody></table></div></details>
  </section>

  <section class="chart">
    <div class="chart-head"><div><h2>How often each tile is reached</h2><p>Share of evaluation games whose best tile was at least 1024, 2048, or 4096.</p></div>
      <div class="legend"><span class="k1">1024</span><span class="k2">2048</span><span class="k3">4096</span></div></div>
    <div class="plot" id="rates"></div>
    <details><summary>Table view</summary><p>Same values as the learning-curve table above.</p></details>
  </section>

  <section class="chart">
    <div class="chart-head"><div><h2>Where {{N_GAMES}} games ended</h2><p>Best tile on the board when the final evaluation games ran out of moves.</p></div></div>
    <div class="plot" id="hist"></div>
    <details><summary>Table view</summary><div class="table-wrap"><table>
      <thead><tr><th>Best tile</th><th>Games</th><th>Share</th></tr></thead>
      <tbody>{{HIST_ROWS}}</tbody></table></div></details>
  </section>

  <section class="method">
    <h2>Method</h2>
    <p>The network scores an <em>afterstate</em>: the board right after a move and before the random tile appears. To play, it tries the four moves and takes the one with the highest immediate reward plus predicted value. The random spawn is kept out of the decision, which is what makes this formulation work well for 2048.</p>
    <p>Learning is afterstate TD(0), the update used with n-tuple networks by Szubert and Jaśkowski, here with a small convolutional net: one-hot tiles, two 2×2 convolutions, and a two-layer head. Hundreds of boards play in parallel with a lookup-table engine; each afterstate is regressed toward the next move's reward plus a slowly tracking target network's value of the next afterstate, or zero when the game ended. Each training sample is passed through a random one of the board's eight symmetries. At play time the value is averaged over all eight.</p>
    <dl>{{CONFIG}}</dl>
  </section>
</div>

<script id="data" type="application/json">{{DATA}}</script>
<script>
(() => {
  const data = JSON.parse(document.getElementById("data").textContent);
  const SVG = "http://www.w3.org/2000/svg";
  const el = (tag, attrs = {}, parent) => {
    const n = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (parent) parent.appendChild(n);
    return n;
  };
  const fmtInt = (n) => Math.round(n).toLocaleString("en-US");
  const fmtPct = (x) => (x >= 0.1 || x === 0 ? Math.round(x * 100) + "%" : (x * 100).toFixed(1) + "%");
  const fmtM = (s) => (s / 1e6).toFixed(s >= 10e6 ? 0 : 1) + "M";
  const niceStep = (span, n) => {
    const raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    for (const m of [1, 2, 2.5, 5, 10]) if (raw <= m * mag) return m * mag;
    return 10 * mag;
  };
  const ticks = (max, n) => { const s = niceStep(max, n); const out = []; for (let v = 0; v <= max + 1e-9; v += s) out.push(v); return out; };

  function tooltip(plot) {
    const t = document.createElement("div"); t.className = "tip"; plot.appendChild(t);
    return {
      show(px, py, xLabel, rows) {
        t.replaceChildren();
        const x = document.createElement("div"); x.className = "x"; x.textContent = xLabel; t.appendChild(x);
        for (const r of rows) {
          const d = document.createElement("div"); d.className = "row";
          const name = document.createElement("span");
          if (r.cls) { const k = document.createElement("i"); k.className = "k"; k.style.background = r.color; name.appendChild(k); }
          name.appendChild(document.createTextNode(r.label));
          const b = document.createElement("b"); b.textContent = r.value;
          d.append(name, b); t.appendChild(d);
        }
        const W = plot.clientWidth, tw = t.offsetWidth || 160;
        t.style.left = Math.min(Math.max(px + 14, 4), W - tw - 4) + "px";
        t.style.top = Math.max(py - 10, 4) + "px";
        t.classList.add("on");
      },
      hide() { t.classList.remove("on"); },
    };
  }

  function lineChart(plot, { xs, series, yMax, yFmt, xTitle, yTicksN = 4, area = false, xFmt = fmtM, xUnit = " moves" }) {
    const W = 880, H = 320, m = { t: 20, r: 78, b: 44, l: 64 };
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, plot);
    const x0 = xs[0], x1 = xs[xs.length - 1];
    const sx = (v) => m.l + ((v - x0) / Math.max(x1 - x0, 1)) * (W - m.l - m.r);
    const sy = (v) => H - m.b - (v / yMax) * (H - m.t - m.b);
    for (const v of ticks(yMax, yTicksN)) {
      el("line", { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), class: v === 0 ? "baseline" : "grid" }, svg);
      const t = el("text", { x: m.l - 10, y: sy(v) + 4, class: "yt" }, svg); t.textContent = yFmt(v);
    }
    for (const v of ticks(x1, 5)) {
      if (v < x0) continue;
      const t = el("text", { x: sx(v), y: H - m.b + 18, class: "xt" }, svg); t.textContent = xFmt(v);
    }
    const at = el("text", { x: W - m.r, y: H - 6, class: "axis-title", "text-anchor": "end" }, svg); at.textContent = xTitle;
    if (area && series.length === 1) {
      const ys = series[0].values;
      el("path", { class: "area", d: "M" + xs.map((x, i) => `${sx(x)},${sy(ys[i])}`).join("L") + `L${sx(x1)},${sy(0)}L${sx(x0)},${sy(0)}Z` }, svg);
    }
    series.forEach((s, si) => {
      el("path", { class: `line l${si + 1}`, d: "M" + xs.map((x, i) => `${sx(x)},${sy(s.values[i])}`).join("L") }, svg);
      const last = s.values[s.values.length - 1];
      el("circle", { cx: sx(x1), cy: sy(last), r: 4.5, class: `dot d${si + 1}` }, svg);
      const t = el("text", { x: sx(x1) + 10, y: sy(last) + 4, class: "end" }, svg); t.textContent = yFmt(last);
    });
    const cross = el("line", { x1: 0, x2: 0, y1: m.t, y2: H - m.b, class: "cross" }, svg);
    const dots = series.map((_, si) => el("circle", { r: 4.5, class: `dot d${si + 1}`, opacity: 0 }, svg));
    const ov = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent", class: "overlay", tabindex: 0 }, svg);
    const tip = tooltip(plot);
    const colors = ["--s1", "--s2", "--s3"].map((v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim());
    let idx = -1;
    const showIdx = (i) => {
      idx = i; const x = sx(xs[i]);
      cross.setAttribute("x1", x); cross.setAttribute("x2", x); cross.style.opacity = 1;
      series.forEach((s, si) => { dots[si].setAttribute("cx", x); dots[si].setAttribute("cy", sy(s.values[i])); dots[si].setAttribute("opacity", 1); });
      const r = svg.getBoundingClientRect(), k = r.width / W;
      tip.show(x * k, sy(series[0].values[i]) * k, xFmt(xs[i]) + xUnit, series.map((s, si) => ({ label: s.label, value: yFmt(s.values[i]), cls: true, color: colors[si] })));
    };
    const hide = () => { cross.style.opacity = 0; dots.forEach((d) => d.setAttribute("opacity", 0)); tip.hide(); idx = -1; };
    ov.addEventListener("pointermove", (e) => {
      const r = svg.getBoundingClientRect(); const vx = ((e.clientX - r.left) / r.width) * W;
      let best = 0; for (let i = 1; i < xs.length; i++) if (Math.abs(sx(xs[i]) - vx) < Math.abs(sx(xs[best]) - vx)) best = i;
      showIdx(best);
    });
    ov.addEventListener("pointerleave", hide);
    ov.addEventListener("focus", () => showIdx(xs.length - 1));
    ov.addEventListener("blur", hide);
    ov.addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft") { e.preventDefault(); showIdx(Math.max(0, (idx < 0 ? xs.length : idx) - 1)); }
      if (e.key === "ArrowRight") { e.preventDefault(); showIdx(Math.min(xs.length - 1, idx + 1)); }
    });
  }

  // total: values are counts out of `total` (shown as shares); otherwise raw values.
  // kind: "tile" draws 2048-style tile chips as category labels; "text" plain labels.
  function barChart(plot, { cats, values, total = null, kind = "tile", rows = null }) {
    const W = 880, H = 300, m = { t: 30, r: 24, b: 56, l: 64 };
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, plot);
    const vals = total ? values.map((v) => v / total) : values;
    const fmt = total ? fmtPct : fmtInt;
    const yMax = Math.max(...vals);
    const band = (W - m.l - m.r) / cats.length, bw = Math.min(24, band * 0.5);
    const sy = (v) => H - m.b - (v / yMax) * (H - m.t - m.b);
    for (const v of ticks(yMax, 4)) {
      el("line", { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), class: v === 0 ? "baseline" : "grid" }, svg);
      const t = el("text", { x: m.l - 10, y: sy(v) + 4, class: "yt" }, svg); t.textContent = fmt(v);
    }
    const tip = tooltip(plot);
    const tileColor = { 256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e" };
    cats.forEach((c, i) => {
      const cx = m.l + band * (i + 0.5), v = vals[i], y = sy(v), h = sy(0) - y, r = Math.min(4, h);
      const d = `M${cx - bw / 2},${sy(0)}V${y + r}a${r},${r} 0 0 1 ${r},-${r}H${cx + bw / 2 - r}a${r},${r} 0 0 1 ${r},${r}V${sy(0)}Z`;
      const bar = el("path", { d, class: "bar" }, svg);
      const cap = el("text", { x: cx, y: y - 8, class: "cap" }, svg); cap.textContent = fmt(v);
      if (kind === "tile") {
        const bg = tileColor[c] || (c >= 4096 ? "#3c3a32" : "#eee4da");
        el("rect", { x: cx - 24, y: H - m.b + 12, width: 48, height: 24, fill: bg, class: "tilebox" }, svg);
        const lbl = el("text", { x: cx, y: H - m.b + 28, class: "tilelbl", fill: c >= 8 ? "#f9f6f2" : "#776e65" }, svg); lbl.textContent = c;
      } else {
        const lbl = el("text", { x: cx, y: H - m.b + 28, class: "xt" }, svg); lbl.textContent = c;
      }
      const hit = el("rect", { x: cx - band / 2, y: m.t, width: band, height: H - m.t - m.b + 40, fill: "transparent", tabindex: 0, class: "overlay" }, svg);
      const tipRows = rows ? rows[i] : (total ? [{ label: "games", value: `${values[i]} of ${total}` }, { label: "share", value: fmtPct(v) }] : [{ label: "value", value: fmtInt(v) }]);
      const show = () => { bar.classList.add("hot"); const k = svg.getBoundingClientRect().width / W; tip.show(cx * k, y * k, (kind === "tile" ? "best tile " : "") + c, tipRows); };
      const hide = () => { bar.classList.remove("hot"); tip.hide(); };
      hit.addEventListener("pointerenter", show); hit.addEventListener("pointerleave", hide);
      hit.addEventListener("focus", show); hit.addEventListener("blur", hide);
    });
  }

  const log = data.log;
  if (log.length) {
    const xs = log.map((r) => r.env_steps);
    const scores = log.map((r) => r.eval_mean_score);
    lineChart(document.getElementById("curve"), {
      xs, series: [{ label: "mean score", values: scores }], area: true,
      yMax: Math.max(...scores) * 1.08, yFmt: fmtInt, xTitle: "moves played",
    });
    lineChart(document.getElementById("rates"), {
      xs, series: [
        { label: "reached 1024", values: log.map((r) => r.rate_1024) },
        { label: "reached 2048", values: log.map((r) => r.rate_2048) },
        { label: "reached 4096", values: log.map((r) => r.rate_4096) },
      ],
      yMax: 1, yFmt: fmtPct, xTitle: "moves played",
    });
  }
  const counts = Object.entries(data.final.tile_counts || {}).map(([k, v]) => [Number(k), v]).sort((a, b) => a[0] - b[0]);
  if (counts.length) {
    barChart(document.getElementById("hist"), { cats: counts.map((c) => c[0]), values: counts.map((c) => c[1]), total: data.final.games });
  }
})();
</script>
"""

NTUPLE_TEMPLATE = r"""<title>2048 N-Tuple Player</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Chivo:wght@400;700;900&family=Source+Sans+3:wght@400;600&display=swap">
<style>
:root {
  color-scheme: light;
  --ground: #fbf8f1; --surface: #fffdf8; --ink: #1f1b16; --ink-2: #5c554b; --muted: #8a8276;
  --grid: #ebe5d9; --baseline: #cfc7b8; --ring: rgba(31,27,22,.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --accent: #edc22e;
  --tip-bg: #1f1b16; --tip-ink: #fbf8f1;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --ground: #1c1a15; --surface: #242119; --ink: #f3eee4; --ink-2: #c9c1b2; --muted: #8f887b;
    --grid: #33302a; --baseline: #46423a; --ring: rgba(243,238,228,.12);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70;
    --tip-bg: #f3eee4; --tip-ink: #1c1a15;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --ground: #1c1a15; --surface: #242119; --ink: #f3eee4; --ink-2: #c9c1b2; --muted: #8f887b;
  --grid: #33302a; --baseline: #46423a; --ring: rgba(243,238,228,.12);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70;
  --tip-bg: #f3eee4; --tip-ink: #1c1a15;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--ground); color: var(--ink);
  font-family: "Source Sans 3", "Helvetica Neue", Arial, sans-serif; font-size: 17px; line-height: 1.5;
  padding-block: 40px 72px; padding-inline: 20px;
}
h1, h2, .figure, .kpi .value { font-family: "Chivo", "Helvetica Neue", Arial, sans-serif; }
h1 { font-size: 2rem; font-weight: 900; letter-spacing: -.01em; margin: 0; text-wrap: balance; line-height: 1.1; }
h2 { font-size: 1.15rem; font-weight: 700; margin: 0 0 4px; }
p { max-width: 66ch; }
.wrap { max-width: 920px; margin: 0 auto; display: flex; flex-direction: column; gap: 40px; }
.eyebrow { font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); font-weight: 600; }
header .meta { color: var(--ink-2); margin-top: 8px; }
header .meta span + span::before { content: "·"; margin: 0 .5em; color: var(--muted); }

.hero { display: flex; align-items: center; gap: 28px; flex-wrap: wrap; }
.figure { font-size: clamp(64px, 12vw, 112px); font-weight: 900; line-height: .95; letter-spacing: -.03em; }
.hero .lede { max-width: 30ch; color: var(--ink-2); font-size: 1.1rem; margin: 0; }
.hero .lede strong { color: var(--ink); }
.note { margin: -24px 0 0; color: var(--ink-2); font-size: .95rem; }

.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.kpi { background: var(--surface); border: 1px solid var(--ring); border-radius: 6px; padding: 14px 16px; }
.kpi .label { font-size: .8rem; color: var(--muted); font-weight: 600; letter-spacing: .02em; }
.kpi .value { font-size: 1.7rem; font-weight: 700; line-height: 1.2; margin-top: 4px; }
.kpi .sub { font-size: .85rem; color: var(--ink-2); }

.tile {
  display: inline-flex; align-items: center; justify-content: center; vertical-align: middle;
  min-width: 2.6em; height: 2.6em; padding: 0 .3em; border-radius: 4px; font-family: "Chivo", sans-serif;
  font-weight: 900; font-size: .8em; color: #776e65; background: #eee4da;
}
.tile[data-v="4"] { background: #ede0c8; }
.tile[data-v="8"] { background: #f2b179; color: #f9f6f2; }
.tile[data-v="16"] { background: #f59563; color: #f9f6f2; }
.tile[data-v="32"] { background: #f67c5f; color: #f9f6f2; }
.tile[data-v="64"] { background: #f65e3b; color: #f9f6f2; }
.tile[data-v="128"] { background: #edcf72; color: #f9f6f2; }
.tile[data-v="256"] { background: #edcc61; color: #f9f6f2; }
.tile[data-v="512"] { background: #edc850; color: #f9f6f2; }
.tile[data-v="1024"] { background: #edc53f; color: #f9f6f2; }
.tile[data-v="2048"] { background: #edc22e; color: #f9f6f2; }
.tile[data-v="4096"], .tile[data-v="8192"], .tile[data-v="16384"], .tile[data-v="32768"], .tile[data-v="65536"] { background: #3c3a32; color: #f9f6f2; }
.hero .tile { font-size: 2.2rem; box-shadow: 0 0 34px 6px rgba(237,194,46,.35); }

section.chart { display: flex; flex-direction: column; gap: 10px; }
.chart-head { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; flex-wrap: wrap; }
.chart-head p { margin: 0; color: var(--ink-2); font-size: .95rem; }
.legend { display: flex; gap: 18px; font-size: .85rem; color: var(--ink-2); flex-wrap: wrap; }
.legend span::before { content: ""; display: inline-block; width: 18px; height: 2px; vertical-align: middle; margin-right: 6px; border-radius: 1px; }
.legend .k1::before { background: var(--s1); } .legend .k2::before { background: var(--s2); } .legend .k3::before { background: var(--s3); }
.plot { position: relative; background: var(--surface); border: 1px solid var(--ring); border-radius: 6px; padding: 8px 4px 2px; }
.plot svg { width: 100%; height: auto; display: block; font-family: "Source Sans 3", sans-serif; }
.plot .overlay:focus { outline: none; }
.plot .overlay:focus-visible + .focus-ring { stroke: var(--s1); }
svg text { fill: var(--ink-2); font-size: 12px; font-variant-numeric: tabular-nums; }
svg .grid { stroke: var(--grid); stroke-width: 1; }
svg .baseline { stroke: var(--baseline); stroke-width: 1; }
svg .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
svg .l1 { stroke: var(--s1); } svg .l2 { stroke: var(--s2); } svg .l3 { stroke: var(--s3); }
svg .area { fill: var(--s1); opacity: .1; }
svg .dot { stroke: var(--surface); stroke-width: 2; }
svg .d1 { fill: var(--s1); } svg .d2 { fill: var(--s2); } svg .d3 { fill: var(--s3); }
svg .end { fill: var(--ink); font-weight: 600; }
svg .cross { stroke: var(--muted); stroke-width: 1; opacity: 0; }
svg .bar { fill: var(--s1); }
svg .bar.hot { opacity: .8; }
svg .cap { fill: var(--ink); font-weight: 600; text-anchor: middle; }
svg .xt { text-anchor: middle; } svg .yt { text-anchor: end; }
svg .axis-title { fill: var(--muted); font-size: 11px; letter-spacing: .04em; text-transform: uppercase; }
svg .tilebox { rx: 3; }
svg .tilelbl { font-family: "Chivo", sans-serif; font-weight: 900; text-anchor: middle; font-size: 11px; }
.tip {
  position: absolute; pointer-events: none; background: var(--tip-bg); color: var(--tip-ink);
  border-radius: 5px; padding: 8px 10px; font-size: .85rem; line-height: 1.35; opacity: 0; transition: opacity .08s;
  min-width: 140px; z-index: 2;
}
.tip.on { opacity: 1; }
.tip .x { color: inherit; opacity: .7; font-size: .78rem; margin-bottom: 2px; }
.tip .row { display: flex; justify-content: space-between; gap: 14px; }
.tip .row b { font-variant-numeric: tabular-nums; }
.tip .row .k { display: inline-block; width: 12px; height: 2px; vertical-align: middle; margin-right: 6px; }
details { font-size: .9rem; color: var(--ink-2); }
summary { cursor: pointer; font-weight: 600; }
table { border-collapse: collapse; margin-top: 8px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 4px 12px 4px 0; border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-size: .8rem; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }
.table-wrap { overflow-x: auto; }
.method p { margin: 0 0 12px; }
dl { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px 20px; margin: 16px 0 0; }
dl div { display: flex; flex-direction: column; gap: 2px; padding: 8px 0; border-top: 1px solid var(--grid); }
dt { font-size: .8rem; color: var(--muted); font-weight: 600; }
dd { margin: 0; font-weight: 600; font-variant-numeric: tabular-nums; }
@media (prefers-reduced-motion: reduce) { .tip { transition: none; } }
@media (max-width: 480px) { body { font-size: 16px; } .hero { gap: 18px; } }
</style>
<div class="wrap">
  <header>
    <div class="eyebrow">Training report</div>
    <h1>An n-tuple network that plays 2048</h1>
    <div class="meta"><span>{{DATE}}</span><span>{{DURATION}} of training</span><span>{{MOVES}} moves</span><span>{{GAMES}} training games</span><span>{{THREADS}} threads at {{MPS}} moves/s</span></div>
  </header>

  <section class="hero">
    <div class="figure">{{HERO_MEAN}}</div>
    {{BEST_TILE}}
    <p class="lede">mean score over <strong>{{HERO_GAMES}} games</strong> with depth-{{HERO_DEPTH}} expectimax on top of the learned tables. {{HERO_RATES}}</p>
  </section>

  <section class="kpis">
    <div class="kpi"><div class="label">Reached 8192</div><div class="value">{{R8192}}</div><div class="sub">of games</div></div>
    <div class="kpi"><div class="label">Reached 16384</div><div class="value">{{R16384}}</div><div class="sub">of games</div></div>
    <div class="kpi"><div class="label">Reached 32768</div><div class="value">{{R32768}}</div><div class="sub">of games</div></div>
    <div class="kpi"><div class="label">Best game</div><div class="value">{{MAX}}</div><div class="sub">points</div></div>
    <div class="kpi"><div class="label">Average game</div><div class="value">{{LENGTH}}</div><div class="sub">moves</div></div>
  </section>

  <section class="chart">
    <div class="chart-head"><div><h2>Search depth</h2><p>Mean score of the same tables at each expectimax depth. Depth 0 takes the move with the best immediate reward plus table value; each level adds a layer of random spawns.</p></div></div>
    <div class="plot" id="depths"></div>
    <details><summary>Table view</summary><div class="table-wrap"><table>
      <thead><tr><th>Depth</th><th>Games</th><th>Mean</th><th>Median</th><th>Best</th><th>8192</th><th>16384</th><th>32768</th></tr></thead>
      <tbody>{{DEPTH_ROWS}}</tbody></table></div></details>
  </section>

  <section class="chart">
    <div class="chart-head"><div><h2>Learning curve</h2><p>Mean score of the training games themselves (greedy play, no search), per chunk of {{CHUNK}} games.</p></div></div>
    <div class="plot" id="curve"></div>
    <details><summary>Table view</summary><div class="table-wrap"><table>
      <thead><tr><th>Games</th><th>Moves</th><th>Mean</th><th>Median</th><th>Best</th><th>8192</th><th>16384</th><th>32768</th></tr></thead>
      <tbody>{{CURVE_ROWS}}</tbody></table></div></details>
  </section>

  <section class="chart">
    <div class="chart-head"><div><h2>How often training games reach each tile</h2><p>Share of greedy training games whose best tile was at least 8192, 16384, or 32768.</p></div>
      <div class="legend"><span class="k1">8192</span><span class="k2">16384</span><span class="k3">32768</span></div></div>
    <div class="plot" id="rates"></div>
    <details><summary>Table view</summary><p>Same values as the learning-curve table above.</p></details>
  </section>

  <section class="chart">
    <div class="chart-head"><div><h2>Where {{HERO_GAMES}} games ended</h2><p>Best tile on the board when the depth-{{HERO_DEPTH}} evaluation games ran out of moves.</p></div></div>
    <div class="plot" id="hist"></div>
    <details><summary>Table view</summary><div class="table-wrap"><table>
      <thead><tr><th>Best tile</th><th>Games</th><th>Share</th></tr></thead>
      <tbody>{{HIST_ROWS}}</tbody></table></div></details>
  </section>

  <section class="method">
    <h2>Method</h2>
    <p>The evaluator is an <em>n-tuple network</em>: {{K}} fixed patterns of {{CELLS}} cells, each looked up under all eight board symmetries, so a board is scored by adding {{LOOKUPS}} table entries. With 16 possible tile values per cell that is {{WEIGHTS}} weights and no arithmetic beyond the additions, which is why one move costs microseconds and why this family holds every strong published 2048 result.</p>
    <p>Learning is afterstate TD(0) self-play in C: after each move the table entries of the previous afterstate are nudged toward the reward just earned plus the value of the new afterstate, with per-entry temporal-coherence step sizes. Twelve threads update the shared tables without locks.{{STAGES_NOTE}}</p>
    <p>At play time, expectimax looks ahead: for each move it averages the best continuation over every possible spawn (a 2 with probability 0.9, a 4 with 0.1, in each empty cell), repeating for the configured depth, with a transposition table so repeated positions are scored once.</p>
    <dl>{{CONFIG}}</dl>
  </section>
</div>
<script id="data" type="application/json">{{DATA}}</script>
<script>
(() => {
  const data = JSON.parse(document.getElementById("data").textContent);
  const SVG = "http://www.w3.org/2000/svg";
  const el = (tag, attrs = {}, parent) => {
    const n = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (parent) parent.appendChild(n);
    return n;
  };
  const fmtInt = (n) => Math.round(n).toLocaleString("en-US");
  const fmtPct = (x) => (x >= 0.1 || x === 0 ? Math.round(x * 100) + "%" : (x * 100).toFixed(1) + "%");
  const fmtM = (s) => (s / 1e6).toFixed(s >= 10e6 ? 0 : 1) + "M";
  const niceStep = (span, n) => {
    const raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    for (const m of [1, 2, 2.5, 5, 10]) if (raw <= m * mag) return m * mag;
    return 10 * mag;
  };
  const ticks = (max, n) => { const s = niceStep(max, n); const out = []; for (let v = 0; v <= max + 1e-9; v += s) out.push(v); return out; };

  function tooltip(plot) {
    const t = document.createElement("div"); t.className = "tip"; plot.appendChild(t);
    return {
      show(px, py, xLabel, rows) {
        t.replaceChildren();
        const x = document.createElement("div"); x.className = "x"; x.textContent = xLabel; t.appendChild(x);
        for (const r of rows) {
          const d = document.createElement("div"); d.className = "row";
          const name = document.createElement("span");
          if (r.cls) { const k = document.createElement("i"); k.className = "k"; k.style.background = r.color; name.appendChild(k); }
          name.appendChild(document.createTextNode(r.label));
          const b = document.createElement("b"); b.textContent = r.value;
          d.append(name, b); t.appendChild(d);
        }
        const W = plot.clientWidth, tw = t.offsetWidth || 160;
        t.style.left = Math.min(Math.max(px + 14, 4), W - tw - 4) + "px";
        t.style.top = Math.max(py - 10, 4) + "px";
        t.classList.add("on");
      },
      hide() { t.classList.remove("on"); },
    };
  }

  function lineChart(plot, { xs, series, yMax, yFmt, xTitle, yTicksN = 4, area = false, xFmt = fmtM, xUnit = " moves" }) {
    const W = 880, H = 320, m = { t: 20, r: 78, b: 44, l: 64 };
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, plot);
    const x0 = xs[0], x1 = xs[xs.length - 1];
    const sx = (v) => m.l + ((v - x0) / Math.max(x1 - x0, 1)) * (W - m.l - m.r);
    const sy = (v) => H - m.b - (v / yMax) * (H - m.t - m.b);
    for (const v of ticks(yMax, yTicksN)) {
      el("line", { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), class: v === 0 ? "baseline" : "grid" }, svg);
      const t = el("text", { x: m.l - 10, y: sy(v) + 4, class: "yt" }, svg); t.textContent = yFmt(v);
    }
    for (const v of ticks(x1, 5)) {
      if (v < x0) continue;
      const t = el("text", { x: sx(v), y: H - m.b + 18, class: "xt" }, svg); t.textContent = xFmt(v);
    }
    const at = el("text", { x: W - m.r, y: H - 6, class: "axis-title", "text-anchor": "end" }, svg); at.textContent = xTitle;
    if (area && series.length === 1) {
      const ys = series[0].values;
      el("path", { class: "area", d: "M" + xs.map((x, i) => `${sx(x)},${sy(ys[i])}`).join("L") + `L${sx(x1)},${sy(0)}L${sx(x0)},${sy(0)}Z` }, svg);
    }
    series.forEach((s, si) => {
      el("path", { class: `line l${si + 1}`, d: "M" + xs.map((x, i) => `${sx(x)},${sy(s.values[i])}`).join("L") }, svg);
      const last = s.values[s.values.length - 1];
      el("circle", { cx: sx(x1), cy: sy(last), r: 4.5, class: `dot d${si + 1}` }, svg);
      const t = el("text", { x: sx(x1) + 10, y: sy(last) + 4, class: "end" }, svg); t.textContent = yFmt(last);
    });
    const cross = el("line", { x1: 0, x2: 0, y1: m.t, y2: H - m.b, class: "cross" }, svg);
    const dots = series.map((_, si) => el("circle", { r: 4.5, class: `dot d${si + 1}`, opacity: 0 }, svg));
    const ov = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent", class: "overlay", tabindex: 0 }, svg);
    const tip = tooltip(plot);
    const colors = ["--s1", "--s2", "--s3"].map((v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim());
    let idx = -1;
    const showIdx = (i) => {
      idx = i; const x = sx(xs[i]);
      cross.setAttribute("x1", x); cross.setAttribute("x2", x); cross.style.opacity = 1;
      series.forEach((s, si) => { dots[si].setAttribute("cx", x); dots[si].setAttribute("cy", sy(s.values[i])); dots[si].setAttribute("opacity", 1); });
      const r = svg.getBoundingClientRect(), k = r.width / W;
      tip.show(x * k, sy(series[0].values[i]) * k, xFmt(xs[i]) + xUnit, series.map((s, si) => ({ label: s.label, value: yFmt(s.values[i]), cls: true, color: colors[si] })));
    };
    const hide = () => { cross.style.opacity = 0; dots.forEach((d) => d.setAttribute("opacity", 0)); tip.hide(); idx = -1; };
    ov.addEventListener("pointermove", (e) => {
      const r = svg.getBoundingClientRect(); const vx = ((e.clientX - r.left) / r.width) * W;
      let best = 0; for (let i = 1; i < xs.length; i++) if (Math.abs(sx(xs[i]) - vx) < Math.abs(sx(xs[best]) - vx)) best = i;
      showIdx(best);
    });
    ov.addEventListener("pointerleave", hide);
    ov.addEventListener("focus", () => showIdx(xs.length - 1));
    ov.addEventListener("blur", hide);
    ov.addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft") { e.preventDefault(); showIdx(Math.max(0, (idx < 0 ? xs.length : idx) - 1)); }
      if (e.key === "ArrowRight") { e.preventDefault(); showIdx(Math.min(xs.length - 1, idx + 1)); }
    });
  }

  // total: values are counts out of `total` (shown as shares); otherwise raw values.
  // kind: "tile" draws 2048-style tile chips as category labels; "text" plain labels.
  function barChart(plot, { cats, values, total = null, kind = "tile", rows = null }) {
    const W = 880, H = 300, m = { t: 30, r: 24, b: 56, l: 64 };
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, plot);
    const vals = total ? values.map((v) => v / total) : values;
    const fmt = total ? fmtPct : fmtInt;
    const yMax = Math.max(...vals);
    const band = (W - m.l - m.r) / cats.length, bw = Math.min(24, band * 0.5);
    const sy = (v) => H - m.b - (v / yMax) * (H - m.t - m.b);
    for (const v of ticks(yMax, 4)) {
      el("line", { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), class: v === 0 ? "baseline" : "grid" }, svg);
      const t = el("text", { x: m.l - 10, y: sy(v) + 4, class: "yt" }, svg); t.textContent = fmt(v);
    }
    const tip = tooltip(plot);
    const tileColor = { 256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e" };
    cats.forEach((c, i) => {
      const cx = m.l + band * (i + 0.5), v = vals[i], y = sy(v), h = sy(0) - y, r = Math.min(4, h);
      const d = `M${cx - bw / 2},${sy(0)}V${y + r}a${r},${r} 0 0 1 ${r},-${r}H${cx + bw / 2 - r}a${r},${r} 0 0 1 ${r},${r}V${sy(0)}Z`;
      const bar = el("path", { d, class: "bar" }, svg);
      const cap = el("text", { x: cx, y: y - 8, class: "cap" }, svg); cap.textContent = fmt(v);
      if (kind === "tile") {
        const bg = tileColor[c] || (c >= 4096 ? "#3c3a32" : "#eee4da");
        el("rect", { x: cx - 24, y: H - m.b + 12, width: 48, height: 24, fill: bg, class: "tilebox" }, svg);
        const lbl = el("text", { x: cx, y: H - m.b + 28, class: "tilelbl", fill: c >= 8 ? "#f9f6f2" : "#776e65" }, svg); lbl.textContent = c;
      } else {
        const lbl = el("text", { x: cx, y: H - m.b + 28, class: "xt" }, svg); lbl.textContent = c;
      }
      const hit = el("rect", { x: cx - band / 2, y: m.t, width: band, height: H - m.t - m.b + 40, fill: "transparent", tabindex: 0, class: "overlay" }, svg);
      const tipRows = rows ? rows[i] : (total ? [{ label: "games", value: `${values[i]} of ${total}` }, { label: "share", value: fmtPct(v) }] : [{ label: "value", value: fmtInt(v) }]);
      const show = () => { bar.classList.add("hot"); const k = svg.getBoundingClientRect().width / W; tip.show(cx * k, y * k, (kind === "tile" ? "best tile " : "") + c, tipRows); };
      const hide = () => { bar.classList.remove("hot"); tip.hide(); };
      hit.addEventListener("pointerenter", show); hit.addEventListener("pointerleave", hide);
      hit.addEventListener("focus", show); hit.addEventListener("blur", hide);
    });
  }

  const log = data.log;
  const fmtG = (g) => (g >= 1e6 ? (g / 1e6).toFixed(g >= 10e6 ? 0 : 1) + "M" : (g / 1e3).toFixed(0) + "K");
  const depths = Object.keys(data.evals).map(Number).sort((a, b) => a - b);
  if (depths.length) {
    barChart(document.getElementById("depths"), {
      cats: depths.map((d) => "depth " + d), values: depths.map((d) => data.evals[d].mean_score), kind: "text",
      rows: depths.map((d) => { const e = data.evals[d]; return [
        { label: "mean score", value: fmtInt(e.mean_score) }, { label: "reach 16384", value: fmtPct(e.rate_16384) },
        { label: "reach 32768", value: fmtPct(e.rate_32768) }]; }),
    });
  }
  if (log.length) {
    const xs = log.map((r) => r.games);
    const scores = log.map((r) => r.mean_score);
    lineChart(document.getElementById("curve"), {
      xs, series: [{ label: "training mean score", values: scores }], area: true,
      yMax: Math.max(...scores) * 1.08, yFmt: fmtInt, xTitle: "training games", xFmt: fmtG, xUnit: " games",
    });
    lineChart(document.getElementById("rates"), {
      xs, series: [
        { label: "reached 8192", values: log.map((r) => r.rate_8192) },
        { label: "reached 16384", values: log.map((r) => r.rate_16384) },
        { label: "reached 32768", values: log.map((r) => r.rate_32768) },
      ],
      yMax: 1, yFmt: fmtPct, xTitle: "training games", xFmt: fmtG, xUnit: " games",
    });
  }
  const hero = depths.length ? data.evals[depths[depths.length - 1]] : null;
  if (hero) {
    const counts = Object.entries(hero.tile_counts || {}).map(([k, v]) => [Number(k), v]).sort((a, b) => a[0] - b[0]);
    barChart(document.getElementById("hist"), { cats: counts.map((c) => c[0]), values: counts.map((c) => c[1]), total: hero.games });
  }
})();
</script>
"""


NT_NUMERIC = {"games", "chunk_moves", "moves", "elapsed_s", "moves_per_s", "mean_score", "median_score",
              "max_score", "mean_length", "max_tile", "rate_2048", "rate_4096", "rate_8192", "rate_16384",
              "rate_32768"}


def fmt_big(n) -> str:
    n = float(n)
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return fmt_int(n)


def build_ntuple_report(log_rows: list[dict], evals: dict, meta: dict) -> str:
    rows = [{k: (_num(v) if k in NT_NUMERIC and not isinstance(v, (int, float)) else v) for k, v in r.items()}
            for r in log_rows]
    evals = {str(k): v for k, v in evals.items()}
    depths = sorted(int(d) for d in evals)
    hero_depth = depths[-1] if depths else 0
    hero = evals.get(str(hero_depth), {})
    counts = {int(k): int(v) for k, v in hero.get("tile_counts", {}).items()}
    n_games = int(hero.get("games") or sum(counts.values()) or 1)
    best_tile = max(counts) if counts else int(hero.get("max_tile", 0))
    patterns = meta.get("patterns") or []
    k = len(patterns)
    cells = sorted({len(p) for p in patterns})
    cells_txt = " or ".join(str(c) for c in cells) if cells else "?"
    weights = sum(16 ** len(p) for p in patterns) * (len(meta.get("boundaries") or []) + 1)
    bounds = meta.get("boundaries") or []
    chunk = int(rows[0]["games"]) if rows else 0

    depth_rows = "".join(
        f"<tr><td>{d}</td><td>{evals[str(d)]['games']}</td><td>{fmt_int(evals[str(d)]['mean_score'])}</td>"
        f"<td>{fmt_int(evals[str(d)]['median_score'])}</td><td>{fmt_int(evals[str(d)]['max_score'])}</td>"
        f"<td>{fmt_pct(evals[str(d)]['rate_8192'])}</td><td>{fmt_pct(evals[str(d)]['rate_16384'])}</td>"
        f"<td>{fmt_pct(evals[str(d)]['rate_32768'])}</td></tr>"
        for d in depths
    )
    curve_rows = "".join(
        f"<tr><td>{fmt_int(r['games'])}</td><td>{fmt_big(r['moves'])}</td><td>{fmt_int(r['mean_score'])}</td>"
        f"<td>{fmt_int(r['median_score'])}</td><td>{fmt_int(r['max_score'])}</td><td>{fmt_pct(r['rate_8192'])}</td>"
        f"<td>{fmt_pct(r['rate_16384'])}</td><td>{fmt_pct(r['rate_32768'])}</td></tr>"
        for r in rows
    )
    hist_rows = "".join(
        f"<tr><td>{tile(t)}</td><td>{c}</td><td>{fmt_pct(c / n_games)}</td></tr>" for t, c in sorted(counts.items())
    )
    cfg_items = [
        ("Patterns", f"{k} x {cells_txt} cells" if k else None),
        ("Weights", fmt_int(weights) if weights else None),
        ("Learning", "temporal coherence" if meta.get("tc", True) else "plain TD"),
        ("Stage boundaries", ", ".join(fmt_int(b) for b in bounds) if bounds else "none (single stage)"),
        ("Threads", meta.get("threads")),
        ("Training speed", f"{fmt_big(meta['moves_per_s'])} moves/s" if meta.get("moves_per_s") else None),
        ("Evaluation depths", ", ".join(str(d) for d in depths) if depths else None),
    ]
    cfg_html = "".join(f"<div><dt>{a}</dt><dd>{b}</dd></div>" for a, b in cfg_items if b is not None)
    rates_txt = (f"{fmt_pct(hero.get('rate_8192', 0))} of games reach 8192, "
                 f"{fmt_pct(hero.get('rate_16384', 0))} reach 16384 and "
                 f"{fmt_pct(hero.get('rate_32768', 0))} reach 32768.") if hero else ""
    stages_note = (f" Boards whose tile mass has passed {', '.join(fmt_int(b) for b in bounds)} use their own "
                   f"tables, seeded from the previous stage the first time an entry is touched." if bounds else "")

    data = json.dumps({"log": rows, "evals": evals, "meta": meta})
    page = NTUPLE_TEMPLATE
    for key, val in {
        "DATA": data,
        "DATE": date.today().strftime("%B %-d, %Y"),
        "DURATION": fmt_duration(meta.get("elapsed_s", 0)),
        "MOVES": fmt_big(meta.get("moves", 0)),
        "GAMES": fmt_int(meta.get("games", 0)),
        "THREADS": meta.get("threads", "?"),
        "MPS": fmt_big(meta.get("moves_per_s", 0)),
        "HERO_MEAN": fmt_int(hero.get("mean_score", 0)),
        "HERO_GAMES": fmt_int(n_games),
        "HERO_DEPTH": hero_depth,
        "HERO_RATES": rates_txt,
        "BEST_TILE": tile(best_tile) if best_tile else "",
        "R8192": fmt_pct(hero.get("rate_8192", 0)),
        "R16384": fmt_pct(hero.get("rate_16384", 0)),
        "R32768": fmt_pct(hero.get("rate_32768", 0)),
        "MAX": fmt_int(hero.get("max_score", 0)),
        "LENGTH": fmt_int(hero.get("mean_length", 0)),
        "DEPTH_ROWS": depth_rows,
        "CURVE_ROWS": curve_rows,
        "HIST_ROWS": hist_rows,
        "CHUNK": fmt_int(chunk),
        "K": k,
        "CELLS": cells_txt,
        "LOOKUPS": 8 * k,
        "WEIGHTS": fmt_big(weights),
        "STAGES_NOTE": stages_note,
        "CONFIG": cfg_html,
    }.items():
        page = page.replace("{{" + key + "}}", str(val))
    return page


def ntuple_main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="checkpoints/ntuple/log.csv")
    parser.add_argument("--evals", default="checkpoints/ntuple/evals.json",
                        help="JSON from examples/evaluate_ntuple.py: {evals: {depth: stats}, meta: {...}}")
    parser.add_argument("--out", default="checkpoints/ntuple/report.html")
    args = parser.parse_args(argv)
    with open(args.log, newline="") as f:
        rows = list(csv.DictReader(f))
    payload = json.loads(Path(args.evals).read_text())
    html = build_ntuple_report(rows, payload["evals"], payload.get("meta", {}))
    Path(args.out).write_text(html)
    print(f"wrote {args.out}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="checkpoints/log.csv")
    parser.add_argument("--eval", default="checkpoints/eval.json")
    parser.add_argument("--out", default="checkpoints/report.html")
    parser.add_argument("--plain", help="eval JSON of the same checkpoint without --symmetric, for comparison")
    args = parser.parse_args(argv)
    rows = read_log(args.log)
    final = json.loads(Path(args.eval).read_text())
    meta = final.get("meta", {})
    config = final.get("config") or meta.get("config") or {}
    plain = json.loads(Path(args.plain).read_text()) if args.plain else None
    html = build_report(rows, final, config, trained_on=final.get("device"),
                        best_env_steps=meta.get("env_steps"), plain=plain)
    Path(args.out).write_text(html)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
