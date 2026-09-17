(() => {
  const KEYS = {
    ArrowUp: 0, ArrowRight: 1, ArrowDown: 2, ArrowLeft: 3,
    w: 0, d: 1, s: 2, a: 3, W: 0, D: 1, S: 2, A: 3,
  };

  const $ = (id) => document.getElementById(id);
  const boardEl = $("board");
  const scoreEl = $("score");
  const bestEl = $("best");
  const deltaEl = $("score-delta");
  const overlayEl = $("overlay");
  const statusEl = $("status");
  const agentSel = $("agent");
  const modeSel = $("mode");
  const modeHint = $("mode-hint");
  const MODE_LABELS = {
    random: "Random tiles", kind: "Kind tiles: never ends the game if it can",
    best: "Best tiles: the spawn that helps you most", evil: "Evil tiles: the spawn that hurts you most",
    choose: "Cool mode: you choose every tile",
  };
  const MODE_HINTS = {
    random: "", kind: "Kind mode: a new tile never ends the game while any placement avoids it.",
    best: "Best mode: each new tile is the one that helps you most.", evil: "Evil mode: each new tile is the one that hurts you most.",
    choose: "Cool mode: after each move, click an empty cell and pick 2 or 4 (or press 2 / 4).",
  };
  let awaiting = false;
  let pickCell = null;   // [row, col] the player clicked while a tile is due
  const loadMode = () => { try { return localStorage.getItem("mode2048") || "random"; } catch { return "random"; } };
  const saveMode = (m) => { try { localStorage.setItem("mode2048", m); } catch {} };
  const autoBtn = $("agent-auto");
  const speedEl = $("speed");
  const speedValEl = $("speed-value");
  const turboEl = $("turbo");

  let prevBoard = null;
  let lastState = null;
  let busy = false;
  let autoTimer = null;

  const loadBest = () => { try { return Number(localStorage.getItem("best2048") || 0); } catch { return 0; } };
  const saveBest = (v) => { try { localStorage.setItem("best2048", String(v)); } catch {} };
  let best = loadBest();
  bestEl.textContent = best;

  function render(state) {
    const spawned = spawnedCell(prevBoard, state.board);
    awaiting = !!state.awaiting_tile;
    if (!awaiting) pickCell = null;
    boardEl.classList.toggle("awaiting", awaiting);
    boardEl.replaceChildren(
      ...state.board.flatMap((row, i) => row.map((v, j) => {
        const el = document.createElement("div");
        el.className = "tile";
        el.setAttribute("role", "gridcell");
        if (v) {
          el.dataset.v = v;
          el.textContent = v;
          if (v > 2048) el.classList.add("big");
          if (spawned && spawned[0] === i && spawned[1] === j) el.classList.add("new");
        } else if (awaiting) {
          el.classList.add("placeable");
          el.setAttribute("role", "button");
          el.setAttribute("aria-label", `place a tile at row ${i + 1}, column ${j + 1}`);
          el.tabIndex = 0;
          if (pickCell && pickCell[0] === i && pickCell[1] === j) {
            el.classList.add("picking");
            for (const val of [2, 4]) {
              const b = document.createElement("button");
              b.className = "pick";
              b.textContent = val;
              b.addEventListener("click", (e) => { e.stopPropagation(); placeTile(i, j, val); });
              el.appendChild(b);
            }
          } else {
            el.addEventListener("click", () => { pickCell = [i, j]; render(lastState); });
          }
        }
        return el;
      }))
    );
    lastState = state;
    scoreEl.textContent = state.score;
    if (state.score > best) { best = state.score; saveBest(best); bestEl.textContent = best; }
    overlayEl.hidden = !state.over;
    if (state.over) stopAuto();
    if (state.mode && modeSel.value !== state.mode && [...modeSel.options].some((o) => o.value === state.mode)) modeSel.value = state.mode;
    modeHint.textContent = MODE_HINTS[state.mode] || "";
    prevBoard = state.board;
  }

  // Cells that were empty and are now filled. After a slide there can be
  // several; the spawned tile is only unambiguous when there is exactly one.
  function spawnedCell(before, after) {
    if (!before) return null;
    const cells = [];
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++)
      if (!before[i][j] && after[i][j]) cells.push([i, j]);
    return cells.length === 1 ? cells[0] : null;
  }

  function showDelta(reward) {
    if (!reward) return;
    deltaEl.textContent = `+${reward}`;
    deltaEl.classList.remove("show");
    void deltaEl.offsetWidth; // restart the animation
    deltaEl.classList.add("show");
  }

  async function api(path, body) {
    const res = await fetch(path, body === undefined
      ? {}
      : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new Error(`${res.status} ${detail}`);
    }
    return res.json();
  }

  async function guarded(fn) {
    if (busy) return;
    busy = true;
    statusEl.textContent = "";
    try {
      await fn();
    } catch (e) {
      statusEl.textContent = e.message;
      stopAuto();
    } finally {
      busy = false;
    }
  }

  const move = (dir) => guarded(async () => {
    if (awaiting) { statusEl.textContent = "Place a tile first: click an empty cell."; return; }
    const s = await api("/api/move", { direction: dir });
    render(s);
    showDelta(s.reward);
  });

  const placeTile = (row, col, value) => guarded(async () => {
    render(await api("/api/place", { row, col, value }));
  });

  const newGame = () => guarded(async () => {
    stopAuto();
    prevBoard = null;
    saveMode(modeSel.value);
    render(await api("/api/new", { mode: modeSel.value }));
  });

  const agentStep = () => guarded(async () => {
    const s = await api("/api/agent/step", { agent: agentSel.value });
    render(s);
    showDelta(s.reward);
  });

  // Max speed: the server plays ~100 ms of moves per request and we render once per batch,
  // so a whole game takes minutes instead of the better part of an hour.
  const agentRun = () => guarded(async () => {
    const t0 = performance.now();
    const s = await api("/api/agent/run", { agent: agentSel.value, ms: 100 });
    render(s);
    showDelta(s.reward);
    const secs = Math.max(0.001, (performance.now() - t0) / 1000);
    speedValEl.textContent = Math.round(s.steps / secs);
  });

  function startAuto() {
    if (autoTimer) return;
    autoBtn.classList.add("active");
    autoBtn.textContent = "Stop";
    const tick = async () => {
      if (!autoTimer) return;
      if (turboEl.checked) {
        await agentRun();
        if (autoTimer) autoTimer = setTimeout(tick, 0);
      } else {
        await agentStep();
        if (autoTimer) autoTimer = setTimeout(tick, 1000 / Number(speedEl.value));
      }
    };
    autoTimer = setTimeout(tick, 0);
  }

  function stopAuto() {
    if (!autoTimer) return;
    clearTimeout(autoTimer);
    autoTimer = null;
    autoBtn.classList.remove("active");
    autoBtn.textContent = "Auto-play";
    speedValEl.textContent = speedEl.value;
  }

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    if (awaiting && pickCell && (e.key === "2" || e.key === "4")) {
      e.preventDefault();
      placeTile(pickCell[0], pickCell[1], Number(e.key));
      return;
    }
    const dir = KEYS[e.key];
    if (dir === undefined) return;
    e.preventDefault();
    if (autoTimer) return;
    move(dir);
  });

  // Touch swipes
  let touchStart = null;
  boardEl.addEventListener("touchstart", (e) => { touchStart = [e.touches[0].clientX, e.touches[0].clientY]; }, { passive: true });
  boardEl.addEventListener("touchend", (e) => {
    if (!touchStart) return;
    const dx = e.changedTouches[0].clientX - touchStart[0];
    const dy = e.changedTouches[0].clientY - touchStart[1];
    touchStart = null;
    if (Math.max(Math.abs(dx), Math.abs(dy)) < 24) return;
    move(Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 1 : 3) : (dy > 0 ? 2 : 0));
  });

  $("new-game").addEventListener("click", newGame);
  $("retry").addEventListener("click", newGame);
  $("agent-step").addEventListener("click", agentStep);
  autoBtn.addEventListener("click", () => (autoTimer ? stopAuto() : startAuto()));
  speedEl.addEventListener("input", () => { speedValEl.textContent = speedEl.value; });
  turboEl.addEventListener("change", () => { turboEl.blur(); if (!turboEl.checked) speedValEl.textContent = speedEl.value; });
  modeSel.addEventListener("change", () => { modeSel.blur(); newGame(); });
  agentSel.addEventListener("change", () => agentSel.blur());

  (async () => {
    try {
      const names = await api("/api/agents");
      agentSel.replaceChildren(...names.map((n) => Object.assign(document.createElement("option"), { value: n, textContent: n })));
      const modes = await api("/api/modes");
      modeSel.replaceChildren(...modes.map((m) => Object.assign(document.createElement("option"), { value: m, textContent: MODE_LABELS[m] || m })));
      const state = await api("/api/state");
      const wanted = loadMode();
      if (state.mode !== wanted && modes.includes(wanted)) { modeSel.value = wanted; render(await api("/api/new", { mode: wanted })); }
      else render(state);
    } catch (e) {
      statusEl.textContent = `Could not reach backend: ${e.message}`;
    }
  })();
})();
