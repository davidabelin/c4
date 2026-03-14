(function () {
  "use strict";

  const appBasePath = String(window.__APP_BASE_PATH__ || "").replace(/\/+$/, "");
  const apiBase = `${appBasePath}/api/v1`;
  const difficultySelect = document.getElementById("difficultySelect");
  const agentSelectWrap = document.getElementById("agentSelectWrap");
  const agentSelect = document.getElementById("agentSelect");
  const openingPlayerSelect = document.getElementById("openingPlayerSelect");
  const timerSecondsInput = document.getElementById("timerSecondsInput");
  const undoModeSelect = document.getElementById("undoModeSelect");
  const forecastToggleSelect = document.getElementById("forecastToggleSelect");
  const forecastLookaheadInput = document.getElementById("forecastLookaheadInput");
  const newGameBtn = document.getElementById("newGameBtn");
  const undoBtn = document.getElementById("undoBtn");
  const resetBtn = document.getElementById("resetBtn");
  const gameStatus = document.getElementById("gameStatus");
  const turnTimerStatus = document.getElementById("turnTimerStatus");
  const boardGrid = document.getElementById("boardGrid");
  const columnButtons = document.getElementById("columnButtons");
  const winsEl = document.getElementById("wins");
  const lossesEl = document.getElementById("losses");
  const tiesEl = document.getElementById("ties");
  const roundsEl = document.getElementById("rounds");
  const outcomeBanner = document.getElementById("outcomeBanner");
  const forecastStatus = document.getElementById("forecastStatus");
  const moveLog = document.getElementById("moveLog");

  const rows = 6;
  const cols = 7;
  const difficultyAgent = {
    easy: "adaptive_midrange",
    medium: "alpha_beta_v9",
    hard: "time_boxed_pruner",
  };

  let currentGame = null;
  let currentBoard = Array(rows * cols).fill(0);
  let recentCellIndices = new Set();
  let isBusy = false;
  let timerTick = null;
  let timerRemaining = 0;
  let undoUsed = false;
  let forecastMap = {};
  let forecastRecommendedColumn = null;

  function setStatus(text) {
    gameStatus.textContent = text;
  }

  function setOutcome(text) {
    outcomeBanner.textContent = text;
  }

  function parseNumber(value, fallback) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
      return fallback;
    }
    return parsed;
  }

  function clamp(value, low, high) {
    return Math.max(low, Math.min(high, value));
  }

  function chosenAgent() {
    const mode = String(difficultySelect.value || "easy");
    if (mode === "custom") {
      return String(agentSelect.value || "alpha_beta_v9");
    }
    return difficultyAgent[mode] || "alpha_beta_v9";
  }

  function updateDifficultyUi() {
    const custom = String(difficultySelect.value || "") === "custom";
    agentSelectWrap.classList.toggle("is-hidden", !custom);
  }

  function timerSeconds() {
    return clamp(parseNumber(timerSecondsInput.value, 20), 5, 300);
  }

  function updateUndoButtonState() {
    if (!currentGame) {
      undoBtn.disabled = true;
      return;
    }
    const mode = String(undoModeSelect.value || "unlimited");
    if (mode === "off") {
      undoBtn.disabled = true;
      return;
    }
    if (mode === "single" && undoUsed) {
      undoBtn.disabled = true;
      return;
    }
    undoBtn.disabled = false;
  }

  function setInteractive(enabled) {
    isBusy = !enabled;
    newGameBtn.disabled = !enabled;
    resetBtn.disabled = !enabled || !currentGame;
    updateUndoButtonState();
    Array.from(columnButtons.querySelectorAll("button")).forEach((button) => {
      if (!enabled || !currentGame || currentGame.status !== "active") {
        button.disabled = true;
        return;
      }
      const col = Number(button.getAttribute("data-col"));
      button.disabled = currentBoard[col] !== 0;
    });
  }

  function stopTurnTimer() {
    if (timerTick) {
      window.clearInterval(timerTick);
      timerTick = null;
    }
  }

  function startTurnTimer() {
    stopTurnTimer();
    if (!currentGame || currentGame.status !== "active") {
      turnTimerStatus.textContent = "Move clock: -";
      return;
    }
    timerRemaining = timerSeconds();
    turnTimerStatus.textContent = `Move clock: ${timerRemaining}s`;
    timerTick = window.setInterval(() => {
      timerRemaining -= 1;
      if (timerRemaining <= 0) {
        stopTurnTimer();
        turnTimerStatus.textContent = "Move clock expired. Auto-playing a random legal move...";
        autoPlayRandomMove();
        return;
      }
      turnTimerStatus.textContent = `Move clock: ${timerRemaining}s`;
    }, 1000);
  }

  function boardAt(row, col) {
    return currentBoard[row * cols + col];
  }

  function renderBoard() {
    boardGrid.innerHTML = "";
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const index = row * cols + col;
        const value = boardAt(row, col);
        const cell = document.createElement("div");
        cell.className = "cell";
        if (value === 1) {
          cell.classList.add("player");
        } else if (value === 2) {
          cell.classList.add("ai");
        }
        if (recentCellIndices.has(index)) {
          cell.classList.add("recent");
        }
        const core = document.createElement("span");
        core.className = "cell-core";
        cell.appendChild(core);
        boardGrid.appendChild(cell);
      }
    }
    setInteractive(!isBusy);
  }

  function updateScore(game) {
    winsEl.textContent = String(game.score_player || 0);
    lossesEl.textContent = String(game.score_ai || 0);
    tiesEl.textContent = String(game.score_ties || 0);
    roundsEl.textContent = String(game.rounds_played || 0);
  }

  function outcomeText(outcome) {
    const key = String(outcome || "").toLowerCase();
    if (key === "player") {
      return "You win.";
    }
    if (key === "ai") {
      return "AI wins.";
    }
    if (key === "tie") {
      return "Draw.";
    }
    if (key === "ongoing") {
      return "Game in progress.";
    }
    return "";
  }

  function pushLog(text) {
    const row = document.createElement("li");
    row.textContent = text;
    moveLog.prepend(row);
  }

  function resetLog() {
    moveLog.innerHTML = "";
  }

  function buildColumnButtons() {
    columnButtons.innerHTML = "";
    for (let col = 0; col < cols; col += 1) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "drop-btn";
      button.setAttribute("data-col", String(col));
      const forecast = forecastMap[col];
      if (forecastRecommendedColumn === col) {
        button.classList.add("recommended");
      }
      button.innerHTML = `
        <span class="drop-forecast">${forecast ? forecast.label : "--"}</span>
        <span class="drop-label">\u2193 ${col + 1}</span>
      `;
      button.addEventListener("click", () => {
        playMove(col);
      });
      columnButtons.appendChild(button);
    }
  }

  function safeJson(response) {
    return response.json().catch(() => ({}));
  }

  async function fetchAgents() {
    const response = await fetch(`${apiBase}/agents`);
    const body = await safeJson(response);
    if (!response.ok) {
      throw new Error(body.error || "Failed to fetch agents");
    }
    agentSelect.innerHTML = "";
    body.agents.forEach((agent) => {
      const option = document.createElement("option");
      option.value = agent.name;
      option.textContent = `${agent.name} (${agent.type})`;
      agentSelect.appendChild(option);
    });
    if (!agentSelect.value) {
      agentSelect.value = "alpha_beta_v9";
    }
  }

  function applyGamePayload(game) {
    const nextBoard = Array.isArray(game.board) ? game.board.slice() : Array(rows * cols).fill(0);
    recentCellIndices = new Set();
    for (let index = 0; index < rows * cols; index += 1) {
      const before = Number(currentBoard[index] || 0);
      const after = Number(nextBoard[index] || 0);
      if (before !== after && after !== 0) {
        recentCellIndices.add(index);
      }
    }
    currentGame = game;
    currentBoard = nextBoard;
    updateScore(game);
    renderBoard();
    buildColumnButtons();
    resetBtn.disabled = false;
    updateUndoButtonState();
  }

  function forecastEnabled() {
    return String(forecastToggleSelect.value || "on") === "on";
  }

  function forecastLookahead() {
    return clamp(parseNumber(forecastLookaheadInput.value, 4), 1, 10);
  }

  function clearForecasts(message) {
    forecastMap = {};
    forecastRecommendedColumn = null;
    if (forecastStatus) {
      forecastStatus.textContent = message || "Forecasts use a heuristic lookahead estimate.";
    }
    buildColumnButtons();
  }

  async function fetchForecasts() {
    if (!currentGame || currentGame.status !== "active") {
      clearForecasts("Forecasts available during active games.");
      return;
    }
    if (!forecastEnabled()) {
      clearForecasts("Forecasts hidden.");
      return;
    }
    try {
      forecastStatus.textContent = "Calculating column forecasts...";
      const params = new URLSearchParams({
        lookahead: String(forecastLookahead()),
        samples: "12",
      });
      const response = await fetch(`${apiBase}/games/${currentGame.game_id}/analysis?${params.toString()}`);
      const body = await safeJson(response);
      if (!response.ok) {
        forecastStatus.textContent = `Forecasts unavailable: ${body.error || "unknown error"}`;
        return;
      }
      forecastMap = {};
      const forecasts = Array.isArray(body.analysis.forecasts) ? body.analysis.forecasts : [];
      forecasts.forEach((entry) => {
        forecastMap[Number(entry.column)] = entry;
      });
      forecastRecommendedColumn = Number.isInteger(body.analysis.recommended_column)
        ? Number(body.analysis.recommended_column)
        : null;
      forecastStatus.textContent = forecasts.length
        ? `Forecasts show heuristic win estimates with ${body.analysis.lookahead}-ply lookahead.`
        : "No forecast available for this position.";
      buildColumnButtons();
      setInteractive(!isBusy);
    } catch (error) {
      forecastStatus.textContent = `Forecasts unavailable: ${String(error)}`;
    }
  }

  async function createGame() {
    if (isBusy) {
      return;
    }
    setInteractive(false);
    stopTurnTimer();
    try {
      const payload = {
        agent: chosenAgent(),
        opening_player: String(openingPlayerSelect.value || "player"),
      };
      const response = await fetch(`${apiBase}/games`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = await safeJson(response);
      if (!response.ok) {
        setStatus(`Failed to create game: ${body.error || "unknown error"}`);
        return;
      }
      undoUsed = false;
      resetLog();
      applyGamePayload(body.game);
      setStatus(`Game ${body.game.game_id} started vs ${body.game.agent_name}.`);
      if (body.opening_move && body.opening_move.actor === "ai") {
        pushLog(`AI opening move: column ${Number(body.opening_move.action) + 1}.`);
        setOutcome("Agent opened the board. Your move.");
      } else {
        setOutcome("Your move.");
      }
      setInteractive(true);
      fetchForecasts().catch(() => null);
      startTurnTimer();
    } catch (error) {
      setStatus(`Failed to create game: ${String(error)}`);
      setInteractive(true);
    }
  }

  function legalColumns() {
    const out = [];
    for (let col = 0; col < cols; col += 1) {
      if (currentBoard[col] === 0) {
        out.push(col);
      }
    }
    return out;
  }

  async function autoPlayRandomMove() {
    if (!currentGame || currentGame.status !== "active" || isBusy) {
      return;
    }
    const legal = legalColumns();
    if (!legal.length) {
      return;
    }
    const chosen = legal[Math.floor(Math.random() * legal.length)];
    await playMove(chosen);
  }

  async function playMove(col) {
    if (!currentGame || isBusy || currentGame.status !== "active") {
      return;
    }
    if (currentBoard[col] !== 0) {
      setStatus(`Column ${col + 1} is full.`);
      return;
    }
    setInteractive(false);
    stopTurnTimer();
    try {
      const response = await fetch(`${apiBase}/games/${currentGame.game_id}/move`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: col }),
      });
      const body = await safeJson(response);
      if (!response.ok) {
        setStatus(`Move failed: ${body.error || "unknown error"}`);
        setInteractive(true);
        startTurnTimer();
        return;
      }
      applyGamePayload(body.game);
      const aiAction = body.turn.ai_action;
      const aiPart = aiAction === null || aiAction === undefined ? "AI: -" : `AI: ${Number(aiAction) + 1}`;
      pushLog(
        `Turn ${Number(body.turn.round_index) + 1} | You: ${Number(body.turn.player_action) + 1} | ${aiPart} | ${body.turn.outcome}`
      );
      const outcome = String(body.turn.outcome || "ongoing");
      const text = outcomeText(outcome);
      if (text) {
        setOutcome(text);
      }
      if (body.game.status === "completed") {
        setStatus(`Game completed. ${text}`);
        clearForecasts("Game complete. Start or reset to view forecasts again.");
        stopTurnTimer();
        setInteractive(true);
        return;
      }
      setStatus(`Played column ${col + 1}.`);
      setInteractive(true);
      fetchForecasts().catch(() => null);
      startTurnTimer();
    } catch (error) {
      setStatus(`Move failed: ${String(error)}`);
      setInteractive(true);
      startTurnTimer();
    }
  }

  async function undoLastTurn() {
    if (!currentGame || isBusy) {
      return;
    }
    const mode = String(undoModeSelect.value || "unlimited");
    if (mode === "off") {
      return;
    }
    if (mode === "single" && undoUsed) {
      return;
    }
    setInteractive(false);
    stopTurnTimer();
    try {
      const response = await fetch(`${apiBase}/games/${currentGame.game_id}/undo`, {
        method: "POST",
      });
      const body = await safeJson(response);
      if (!response.ok) {
        setStatus(`Undo failed: ${body.error || "unknown error"}`);
        setInteractive(true);
        if (currentGame.status === "active") {
          startTurnTimer();
        }
        return;
      }
      applyGamePayload(body.game);
      undoUsed = true;
      updateUndoButtonState();
      pushLog(`Undo applied (${body.undo.kind}).`);
      setOutcome("Board rewound one step.");
      setStatus("Undo complete.");
      setInteractive(true);
      fetchForecasts().catch(() => null);
      if (currentGame.status === "active") {
        startTurnTimer();
      }
    } catch (error) {
      setStatus(`Undo failed: ${String(error)}`);
      setInteractive(true);
      if (currentGame && currentGame.status === "active") {
        startTurnTimer();
      }
    }
  }

  async function resetGame() {
    if (!currentGame || isBusy) {
      return;
    }
    setInteractive(false);
    stopTurnTimer();
    try {
      const response = await fetch(`${apiBase}/games/${currentGame.game_id}/reset`, {
        method: "POST",
      });
      const body = await safeJson(response);
      if (!response.ok) {
        setStatus(`Reset failed: ${body.error || "unknown error"}`);
        setInteractive(true);
        return;
      }
      undoUsed = false;
      resetLog();
      applyGamePayload(body.game);
      setOutcome("Fresh board ready. Your move.");
      setStatus(`Game ${body.game.game_id} reset.`);
      setInteractive(true);
      fetchForecasts().catch(() => null);
      startTurnTimer();
    } catch (error) {
      setStatus(`Reset failed: ${String(error)}`);
      setInteractive(true);
    }
  }

  difficultySelect.addEventListener("change", updateDifficultyUi);
  newGameBtn.addEventListener("click", createGame);
  undoBtn.addEventListener("click", undoLastTurn);
  resetBtn.addEventListener("click", resetGame);
  undoModeSelect.addEventListener("change", updateUndoButtonState);
  forecastToggleSelect.addEventListener("change", () => {
    if (forecastEnabled()) {
      fetchForecasts().catch(() => null);
    } else {
      clearForecasts("Forecasts hidden.");
    }
  });
  forecastLookaheadInput.addEventListener("change", () => {
    if (forecastEnabled()) {
      fetchForecasts().catch(() => null);
    }
  });
  timerSecondsInput.addEventListener("change", () => {
    if (currentGame && currentGame.status === "active" && !isBusy) {
      startTurnTimer();
    }
  });

  buildColumnButtons();
  renderBoard();
  clearForecasts("Forecasts use a heuristic lookahead estimate.");
  setInteractive(true);
  updateDifficultyUi();
  fetchAgents()
    .then(() => {
      setStatus("Ready. Configure options and start a game.");
    })
    .catch((error) => {
      setStatus(`Failed to initialize agents: ${String(error)}`);
    });
})();
