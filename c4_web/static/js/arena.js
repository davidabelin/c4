(function () {
  "use strict";

  const appBasePath = String(window.__APP_BASE_PATH__ || "").replace(/\/+$/, "");
  const apiBase = `${appBasePath}/api/v1`;
  const rows = 6;
  const cols = 7;

  const arenaForm = document.getElementById("arenaForm");
  const agentASelect = document.getElementById("agentASelect");
  const agentBSelect = document.getElementById("agentBSelect");
  const startingAgentSelect = document.getElementById("startingAgentSelect");
  const arenaSpeedSelect = document.getElementById("arenaSpeedSelect");
  const arenaEstimateSelect = document.getElementById("arenaEstimateSelect");
  const arenaLookaheadInput = document.getElementById("arenaLookaheadInput");
  const startArenaBtn = document.getElementById("startArenaBtn");
  const pauseArenaBtn = document.getElementById("pauseArenaBtn");
  const arenaStatus = document.getElementById("arenaStatus");
  const arenaPlaybackStatus = document.getElementById("arenaPlaybackStatus");
  const arenaOutcomeBanner = document.getElementById("arenaOutcomeBanner");
  const arenaForecastStatus = document.getElementById("arenaForecastStatus");
  const arenaProgress = document.getElementById("arenaProgress");
  const arenaMovesShown = document.getElementById("arenaMovesShown");
  const arenaWinner = document.getElementById("arenaWinner");
  const arenaMatchId = document.getElementById("arenaMatchId");
  const boardGrid = document.getElementById("boardGrid");
  const spectatorColumnLabels = document.getElementById("spectatorColumnLabels");
  const moveLog = document.getElementById("moveLog");
  const arenaMatchesTableBody = document.getElementById("arenaMatchesTableBody");
  const refreshArenaMatchesBtn = document.getElementById("refreshArenaMatchesBtn");

  let currentBoard = Array(rows * cols).fill(0);
  let currentTrace = [];
  let currentMatchId = null;
  let renderedCount = 0;
  let playbackTimer = null;
  let paused = false;
  let eventSource = null;
  let recentCellIndices = new Set();
  let currentFrame = null;

  function setStatus(text) {
    arenaStatus.textContent = text;
  }

  function setPlaybackStatus(text) {
    arenaPlaybackStatus.textContent = text;
  }

  function setProgress(progress) {
    const pct = Math.max(0, Math.min(100, Math.round((progress || 0) * 100)));
    arenaProgress.style.width = `${pct}%`;
  }

  function renderBoard() {
    boardGrid.innerHTML = "";
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const index = row * cols + col;
        const value = Number(currentBoard[index] || 0);
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
  }

  function estimatesEnabled() {
    return String(arenaEstimateSelect.value || "on") === "on";
  }

  function buildColumnLabels(frame) {
    const forecastLookup = {};
    const recommendedColumn = frame && Number.isInteger(frame.recommended_column) ? Number(frame.recommended_column) : null;
    const chosenColumn = frame && Number.isInteger(frame.action) ? Number(frame.action) : null;
    if (frame && Array.isArray(frame.forecasts)) {
      frame.forecasts.forEach((entry) => {
        forecastLookup[Number(entry.column)] = entry;
      });
    }
    spectatorColumnLabels.innerHTML = "";
    for (let col = 0; col < cols; col += 1) {
      const label = document.createElement("button");
      label.type = "button";
      label.className = "drop-btn";
      label.disabled = true;
      if (recommendedColumn === col) {
        label.classList.add("recommended");
      }
      if (chosenColumn === col) {
        label.classList.add("chosen");
      }
      const forecast = estimatesEnabled() ? forecastLookup[col] : null;
      label.innerHTML = `
        <span class="drop-forecast">${forecast ? forecast.label : "--"}</span>
        <span class="drop-label">${col + 1}</span>
      `;
      spectatorColumnLabels.appendChild(label);
    }
    if (chosenColumn !== null && chosenColumn >= 0 && chosenColumn < spectatorColumnLabels.children.length) {
      const chosenLabel = spectatorColumnLabels.children[chosenColumn];
      chosenLabel.classList.add("flash-choice");
      window.setTimeout(
        () => chosenLabel.classList.remove("flash-choice"),
        Math.max(180, Number(arenaSpeedSelect.value || 550) * 0.72)
      );
    }
  }

  function clearPlayback() {
    currentBoard = Array(rows * cols).fill(0);
    currentTrace = [];
    currentMatchId = null;
    renderedCount = 0;
    recentCellIndices = new Set();
    currentFrame = null;
    moveLog.innerHTML = "";
    arenaMovesShown.textContent = "0";
    arenaWinner.textContent = "-";
    arenaMatchId.textContent = "-";
    arenaOutcomeBanner.textContent = "Start a match to begin playback.";
    if (arenaForecastStatus) {
      arenaForecastStatus.textContent = "Live frames arrive through the arena event stream.";
    }
    setProgress(0);
    renderBoard();
    buildColumnLabels(null);
  }

  function stopPlaybackTimer() {
    if (playbackTimer) {
      window.clearTimeout(playbackTimer);
      playbackTimer = null;
    }
  }

  function closeEvents() {
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
  }

  function safeJson(response) {
    return response.json().catch(() => ({}));
  }

  async function fetchAgents() {
    const response = await fetch(`${apiBase}/agents`);
    const body = await safeJson(response);
    if (!response.ok) {
      throw new Error(body.error || "Failed to load agents");
    }
    const names = body.agents.map((agent) => agent.name);
    [agentASelect, agentBSelect].forEach((select) => {
      select.innerHTML = "";
      names.forEach((name) => {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        select.appendChild(option);
      });
    });
    if (names.length) {
      agentASelect.value = names[0];
      agentBSelect.value = names[Math.min(1, names.length - 1)];
    }
  }

  function renderFrame(frame) {
    currentFrame = frame;
    const nextBoard = Array.isArray(frame.board_after) ? frame.board_after.slice() : currentBoard.slice();
    recentCellIndices = new Set();
    for (let index = 0; index < nextBoard.length; index += 1) {
      const before = Number(currentBoard[index] || 0);
      const after = Number(nextBoard[index] || 0);
      if (before !== after && after !== 0) {
        recentCellIndices.add(index);
      }
    }
    currentBoard = nextBoard;
    renderBoard();
    buildColumnLabels(frame);
    arenaMovesShown.textContent = String(frame.move_index + 1);
    arenaOutcomeBanner.textContent = `Move ${frame.move_index + 1}: ${frame.actor} dropped in column ${Number(frame.action) + 1}.`;
    if (arenaForecastStatus) {
      if (!estimatesEnabled()) {
        arenaForecastStatus.textContent = "Column estimates hidden for replay.";
      } else if (Array.isArray(frame.forecasts) && frame.forecasts.length) {
        arenaForecastStatus.textContent =
          `Estimated likelihood choosing a column will result in a win for ${frame.actor === "agent_a" ? "Agent A" : "Agent B"} before this move. ${frame.analysis_lookahead || 4}-ply lookahead.`;
      } else {
        arenaForecastStatus.textContent = "No column estimates were recorded for this match.";
      }
    }
    const item = document.createElement("li");
    item.textContent = `Move ${frame.move_index + 1} | ${frame.actor} -> column ${Number(frame.action) + 1} | ${frame.outcome}`;
    moveLog.prepend(item);
  }

  function schedulePlayback() {
    stopPlaybackTimer();
    if (paused || renderedCount >= currentTrace.length) {
      return;
    }
    playbackTimer = window.setTimeout(() => {
      const frame = currentTrace[renderedCount];
      if (frame) {
        renderFrame(frame);
        renderedCount += 1;
      }
      schedulePlayback();
    }, Number(arenaSpeedSelect.value || 550));
  }

  function applyMatchPayload(match, resetPlayback) {
    currentMatchId = match.id;
    currentTrace = Array.isArray(match.trace) ? match.trace.slice() : [];
    arenaMatchId.textContent = String(match.id);
    setStatus(`Arena match ${match.id}: ${match.status}`);
    setProgress(match.progress);
    if (resetPlayback) {
      currentBoard = Array(rows * cols).fill(0);
      renderedCount = 0;
      moveLog.innerHTML = "";
    }
    const summary = match.summary || {};
    if (summary.winner) {
      arenaWinner.textContent = String(summary.winner);
    } else if (match.winner) {
      arenaWinner.textContent = String(match.winner);
    }
    if (match.status === "completed" && renderedCount >= currentTrace.length) {
      arenaOutcomeBanner.textContent = `Match winner: ${match.winner || "tie"}`;
    }
    schedulePlayback();
  }

  async function fetchMatches() {
    const response = await fetch(`${apiBase}/arena/matches`);
    const body = await safeJson(response);
    if (!response.ok) {
      throw new Error(body.error || "Failed to load arena matches");
    }
    arenaMatchesTableBody.innerHTML = "";
    body.matches.forEach((match) => {
      const tr = document.createElement("tr");
      const params = match.params || {};
      tr.innerHTML = `
        <td>${match.id}</td>
        <td>${match.status}</td>
        <td>${match.agent_a} vs ${match.agent_b}</td>
        <td>${match.winner || "-"}</td>
        <td>${params.starting_agent || "-"}</td>
        <td><button class="btn btn-secondary" type="button" data-match-id="${match.id}">Load</button></td>
      `;
      arenaMatchesTableBody.appendChild(tr);
    });
    arenaMatchesTableBody.querySelectorAll("button[data-match-id]").forEach((button) => {
      button.addEventListener("click", () => loadMatch(Number(button.getAttribute("data-match-id"))));
    });
  }

  async function loadMatch(matchId) {
    closeEvents();
    stopPlaybackTimer();
    const response = await fetch(`${apiBase}/arena/matches/${matchId}`);
    const body = await safeJson(response);
    if (!response.ok) {
      setStatus(`Failed to load match ${matchId}: ${body.error || "unknown error"}`);
      return;
    }
    applyMatchPayload(body.match, true);
  }

  function connectEvents(matchId) {
    closeEvents();
    if (!window.EventSource) {
      return;
    }
    eventSource = new EventSource(`${apiBase}/arena/matches/${matchId}/events`);
    eventSource.addEventListener("update", async (event) => {
      const match = JSON.parse(event.data);
      applyMatchPayload(match, false);
      if (match.status === "completed" || match.status === "failed") {
        await fetchMatches();
      }
    });
    eventSource.addEventListener("end", () => {
      closeEvents();
      fetchMatches().catch(() => null);
    });
    eventSource.onerror = () => {
      closeEvents();
    };
  }

  arenaForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearPlayback();
    closeEvents();
    stopPlaybackTimer();
    startArenaBtn.disabled = true;
    const payload = {
      agent_a: String(agentASelect.value || ""),
      agent_b: String(agentBSelect.value || ""),
      starting_agent: String(startingAgentSelect.value || "agent_a"),
      analysis_enabled: estimatesEnabled(),
      analysis_lookahead: Number(arenaLookaheadInput.value || 4),
    };
    const response = await fetch(`${apiBase}/arena/matches`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await safeJson(response);
    startArenaBtn.disabled = false;
    if (!response.ok) {
      setStatus(`Failed to start arena match: ${body.error || "unknown error"}`);
      return;
    }
    paused = false;
    pauseArenaBtn.textContent = "Pause";
    setPlaybackStatus("Streaming live");
    applyMatchPayload(body.match, true);
    connectEvents(body.match.id);
    await fetchMatches();
  });

  pauseArenaBtn.addEventListener("click", () => {
    paused = !paused;
    pauseArenaBtn.textContent = paused ? "Resume" : "Pause";
    setPlaybackStatus(paused ? "Paused" : "Streaming live");
    if (paused) {
      stopPlaybackTimer();
    } else {
      schedulePlayback();
    }
  });

  refreshArenaMatchesBtn.addEventListener("click", () => {
    fetchMatches().catch((error) => setStatus(`Failed to refresh matches: ${String(error)}`));
  });

  arenaEstimateSelect.addEventListener("change", () => {
    buildColumnLabels(currentFrame);
    if (!estimatesEnabled() && arenaForecastStatus) {
      arenaForecastStatus.textContent = "Column estimates hidden for replay.";
    }
  });

  arenaLookaheadInput.addEventListener("change", () => {
    const bounded = Math.max(1, Math.min(6, Number(arenaLookaheadInput.value || 4)));
    arenaLookaheadInput.value = String(bounded);
  });

  buildColumnLabels(null);
  renderBoard();
  fetchAgents()
    .then(fetchMatches)
    .catch((error) => setStatus(`Failed to initialize arena: ${String(error)}`));
})();
