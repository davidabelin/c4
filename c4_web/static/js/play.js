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
  let isBusy = false;
  let timerTick = null;
  let timerRemaining = 0;
  let undoUsed = false;

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
      turnTimerStatus.textContent = "Turn timer: -";
      return;
    }
    timerRemaining = timerSeconds();
    turnTimerStatus.textContent = `Turn timer: ${timerRemaining}s`;
    timerTick = window.setInterval(() => {
      timerRemaining -= 1;
      if (timerRemaining <= 0) {
        stopTurnTimer();
        turnTimerStatus.textContent = "Turn timer expired. Auto-playing a random legal move...";
        autoPlayRandomMove();
        return;
      }
      turnTimerStatus.textContent = `Turn timer: ${timerRemaining}s`;
    }, 1000);
  }

  function boardAt(row, col) {
    return currentBoard[row * cols + col];
  }

  function renderBoard() {
    boardGrid.innerHTML = "";
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const value = boardAt(row, col);
        const cell = document.createElement("div");
        cell.className = "cell";
        if (value === 1) {
          cell.classList.add("player");
        } else if (value === 2) {
          cell.classList.add("ai");
        }
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
      button.textContent = `\u2193 ${col + 1}`;
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
    currentGame = game;
    currentBoard = Array.isArray(game.board) ? game.board.slice() : Array(rows * cols).fill(0);
    updateScore(game);
    renderBoard();
    resetBtn.disabled = false;
    updateUndoButtonState();
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
      }
      setOutcome("Your turn.");
      setInteractive(true);
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
        stopTurnTimer();
        setInteractive(true);
        return;
      }
      setStatus(`Played column ${col + 1}.`);
      setInteractive(true);
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
      setOutcome("Turn undone.");
      setStatus("Undo complete.");
      setInteractive(true);
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
      setOutcome("Board reset. Your turn.");
      setStatus(`Game ${body.game.game_id} reset.`);
      setInteractive(true);
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
  timerSecondsInput.addEventListener("change", () => {
    if (currentGame && currentGame.status === "active" && !isBusy) {
      startTurnTimer();
    }
  });

  buildColumnButtons();
  renderBoard();
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
