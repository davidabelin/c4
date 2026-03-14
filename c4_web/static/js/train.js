(function () {
  "use strict";

  const appBasePath = String(window.__APP_BASE_PATH__ || "").replace(/\/+$/, "");
  const apiBase = `${appBasePath}/api/v1`;

  const trainForm = document.getElementById("trainForm");
  const modelTypeInput = trainForm.querySelector('[name="model_type"]');
  const lookbackInput = trainForm.querySelector('[name="lookback"]');
  const selectionModeInput = trainForm.querySelector('[name="selection_mode"]');
  const actorScopeInput = trainForm.querySelector('[name="actor_scope"]');
  const hiddenLayer1Input = trainForm.querySelector('[name="hidden_layer_1"]');
  const hiddenLayer2Input = trainForm.querySelector('[name="hidden_layer_2"]');
  const batchSizeInput = trainForm.querySelector('[name="batch_size"]');
  const epochsInput = trainForm.querySelector('[name="epochs"]');
  const mlpFields = Array.from(trainForm.querySelectorAll(".mlp-only"));
  const modelDescription = document.getElementById("modelDescription");
  const mlpHint = document.getElementById("mlpHint");
  const readinessStatus = document.getElementById("readinessStatus");
  const sessionTableBody = document.getElementById("sessionTableBody");
  const refreshSessionsBtn = document.getElementById("refreshSessionsBtn");
  const jobStatus = document.getElementById("jobStatus");
  const jobProgress = document.getElementById("jobProgress");
  const jobMetrics = document.getElementById("jobMetrics");
  const jobChartLine = document.getElementById("jobChartLine");
  const modelTableBody = document.getElementById("modelTableBody");
  const refreshModelsBtn = document.getElementById("refreshModelsBtn");

  let activeJobId = null;
  let pollTimer = null;
  let eventSource = null;
  let chartPoints = [];

  const modelDescriptions = {
    decision_tree: "Decision Tree: fast, interpretable baseline for board-state patterns.",
    mlp: "Neural Network: richer nonlinear baseline for board and move-history structure.",
    frequency: "Frequency baseline: predicts from observed board-context frequencies.",
  };

  function setStatus(text) {
    jobStatus.textContent = text;
  }

  function setProgress(value) {
    const pct = Math.max(0, Math.min(100, Math.round((value || 0) * 100)));
    jobProgress.style.width = `${pct}%`;
  }

  function safeJson(response) {
    return response.json().catch(() => ({}));
  }

  function currentDatasetQuery() {
    const params = new URLSearchParams();
    params.set("lookback", String(Number(lookbackInput.value || 5)));
    params.set("selection_mode", String(selectionModeInput.value || "all"));
    params.set("actor_scope", String(actorScopeInput.value || "algorithm"));
    return params;
  }

  function toPayload(formData) {
    const modelType = String(formData.get("model_type") || "decision_tree");
    const isMlp = modelType === "mlp";
    const hidden1 = Number(hiddenLayer1Input.value || 64);
    const hidden2 = Number(hiddenLayer2Input.value || 32);
    const hiddenLayers = isMlp
      ? [hidden1, hidden2].filter((value) => Number.isFinite(value) && value > 0)
      : [64, 32];
    const batchRaw = isMlp ? String(batchSizeInput.value || "").trim() : "auto";
    return {
      model_type: modelType,
      lookback: Number(formData.get("lookback") || 5),
      selection_mode: String(formData.get("selection_mode") || "all"),
      actor_scope: String(formData.get("actor_scope") || "algorithm"),
      learning_rate: Number(formData.get("learning_rate") || 0.001),
      hidden_layer_sizes: hiddenLayers.length ? hiddenLayers : [64, 32],
      batch_size: batchRaw === "" ? "auto" : batchRaw,
      epochs: isMlp ? Number(epochsInput.value || 200) : 200,
      test_size: 0.2,
      random_state: 42,
    };
  }

  function renderMetrics(job) {
    if (!job.metrics) {
      jobMetrics.textContent = "No metrics yet.";
      return;
    }
    jobMetrics.textContent = JSON.stringify(job.metrics, null, 2);
  }

  function updateModelDescription() {
    const modelType = modelTypeInput.value;
    if (modelDescription) {
      modelDescription.textContent = modelDescriptions[modelType] || "Model description unavailable.";
    }
  }

  function updateMlpHint() {
    const isMlp = modelTypeInput.value === "mlp";
    mlpFields.forEach((field) => {
      field.classList.toggle("is-hidden", !isMlp);
      field.setAttribute("aria-hidden", String(!isMlp));
      const input = field.querySelector("input, select, textarea");
      if (input) {
        input.disabled = !isMlp;
      }
    });
    mlpHint.textContent = isMlp
      ? "Neural-network mode: hidden layers, batch size, and epochs are used."
      : "Non-neural-network mode: NN-only settings are hidden and ignored.";
    updateModelDescription();
  }

  async function fetchReadiness() {
    const response = await fetch(`${apiBase}/training/readiness?${currentDatasetQuery().toString()}`);
    const body = await safeJson(response);
    if (!response.ok) {
      readinessStatus.textContent = `Readiness check failed: ${body.error || "unknown error"}`;
      return;
    }
    const info = body.readiness;
    const sessionCount = Number(info.session_count || 0);
    let text = `Samples: ${info.sample_count} (minimum ${info.minimum_required_samples}) from ${sessionCount} sessions.`;
    text += info.can_train ? " Ready to train." : " Need more matching moves.";
    if (!info.sklearn_available) {
      text += ` scikit-learn unavailable: ${info.sklearn_import_error || "import failed"}.`;
    }
    readinessStatus.textContent = text;
  }

  function renderChart() {
    if (!chartPoints.length) {
      jobChartLine.setAttribute("points", "");
      return;
    }
    const width = 300;
    const height = 90;
    const maxIndex = Math.max(1, chartPoints.length - 1);
    const points = chartPoints.map((point, index) => {
      const x = (index / maxIndex) * width;
      const y = height - point * height;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    });
    jobChartLine.setAttribute("points", points.join(" "));
  }

  function pushChartPoint(progressValue) {
    chartPoints.push(Math.max(0, Math.min(1, Number(progressValue || 0))));
    if (chartPoints.length > 50) {
      chartPoints = chartPoints.slice(chartPoints.length - 50);
    }
    renderChart();
  }

  function resetChart() {
    chartPoints = [];
    renderChart();
  }

  function selectionLabel(value) {
    if (value === "include") {
      return "Included";
    }
    if (value === "exclude") {
      return "Excluded";
    }
    return "Default";
  }

  function moveSummary(session) {
    const human = Number(session.human_moves || 0);
    const algorithm = Number(session.algorithm_moves || 0);
    if (session.session_type === "algorithm_vs_algorithm") {
      return `${algorithm} algorithm moves`;
    }
    return `${human} human / ${algorithm} algorithm`;
  }

  async function setSessionSelection(sourceKind, sourceId, sessionIndex, selection) {
    const response = await fetch(`${apiBase}/training/sessions/selection`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_kind: sourceKind,
        source_id: sourceId,
        session_index: sessionIndex,
        selection,
      }),
    });
    const body = await safeJson(response);
    if (!response.ok) {
      throw new Error(body.error || "Failed to update session selection");
    }
    await Promise.all([fetchSessions(), fetchReadiness()]);
  }

  async function deleteSession(sourceKind, sourceId, sessionIndex) {
    const response = await fetch(`${apiBase}/training/sessions`, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_kind: sourceKind,
        source_id: sourceId,
        session_index: sessionIndex,
      }),
    });
    const body = await safeJson(response);
    if (!response.ok) {
      throw new Error(body.error || "Failed to delete session");
    }
    await Promise.all([fetchSessions(), fetchReadiness()]);
  }

  async function fetchSessions() {
    const response = await fetch(`${apiBase}/training/sessions?limit=200`);
    const body = await safeJson(response);
    if (!response.ok) {
      throw new Error(body.error || "Failed to fetch training sessions");
    }
    sessionTableBody.innerHTML = "";
    if (!body.sessions.length) {
      const row = document.createElement("tr");
      row.innerHTML = '<td colspan="5">No recorded sessions yet.</td>';
      sessionTableBody.appendChild(row);
      return;
    }
    body.sessions.forEach((session) => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>
          <strong>${session.label}</strong><br>
          <span class="form-note">${session.matchup_label}</span>
        </td>
        <td>${session.session_type === "algorithm_vs_algorithm" ? "Algorithm vs Algorithm" : "Human vs Algorithm"}</td>
        <td>${moveSummary(session)}</td>
        <td>${selectionLabel(session.selection)}</td>
        <td>
          <div class="controls-row">
            <button class="btn btn-secondary" type="button" data-selection="include">Include</button>
            <button class="btn btn-secondary" type="button" data-selection="exclude">Exclude</button>
            <button class="btn btn-secondary" type="button" data-selection="">Default</button>
            <button class="btn btn-danger" type="button" data-action="delete">Delete</button>
          </div>
        </td>
      `;
      Array.from(row.querySelectorAll("button[data-selection]")).forEach((button) => {
        button.addEventListener("click", async () => {
          button.disabled = true;
          try {
            await setSessionSelection(
              session.source_kind,
              Number(session.source_id),
              Number(session.session_index),
              button.getAttribute("data-selection") || null
            );
          } catch (error) {
            setStatus(`Session update failed: ${String(error)}`);
          } finally {
            button.disabled = false;
          }
        });
      });
      const deleteBtn = row.querySelector('button[data-action="delete"]');
      if (deleteBtn) {
        deleteBtn.addEventListener("click", async () => {
          if (!window.confirm(`Delete ${session.label} from the curated training dataset?`)) {
            return;
          }
          deleteBtn.disabled = true;
          try {
            await deleteSession(session.source_kind, Number(session.source_id), Number(session.session_index));
            setStatus(`Deleted ${session.label}`);
          } catch (error) {
            setStatus(`Session delete failed: ${String(error)}`);
          } finally {
            deleteBtn.disabled = false;
          }
        });
      }
      sessionTableBody.appendChild(row);
    });
  }

  async function fetchModels() {
    const response = await fetch(`${apiBase}/models`);
    const body = await safeJson(response);
    if (!response.ok) {
      throw new Error(body.error || "Failed to fetch models");
    }
    modelTableBody.innerHTML = "";
    body.models.forEach((model) => {
      const tr = document.createElement("tr");
      const accuracy = model.metrics && model.metrics.test_accuracy !== undefined
        ? Number(model.metrics.test_accuracy).toFixed(3)
        : "-";
      const displayType = model.model_type === "mlp" ? "Neural Network" : model.model_type;
      tr.innerHTML = `
        <td>${model.id}</td>
        <td>${model.name}</td>
        <td>${displayType}</td>
        <td>${accuracy}</td>
        <td>${model.is_active ? "yes" : "no"}</td>
        <td><button class="btn btn-secondary" data-model-id="${model.id}">Activate</button></td>
      `;
      modelTableBody.appendChild(tr);
    });
    Array.from(modelTableBody.querySelectorAll("button[data-model-id]")).forEach((button) => {
      button.addEventListener("click", async () => {
        const modelId = Number(button.getAttribute("data-model-id"));
        await activateModel(modelId);
      });
    });
  }

  async function activateModel(modelId) {
    const response = await fetch(`${apiBase}/models/${modelId}/activate`, { method: "POST" });
    const body = await safeJson(response);
    if (!response.ok) {
      throw new Error(body.error || "Activation failed");
    }
    setStatus(`Activated model ${body.model.name}`);
    await fetchModels();
  }

  async function pollJobOnce() {
    if (!activeJobId) {
      return;
    }
    const response = await fetch(`${apiBase}/training/jobs/${activeJobId}`);
    const body = await safeJson(response);
    if (!response.ok) {
      setStatus(`Job poll failed: ${body.error || "unknown error"}`);
      return;
    }
    const job = body.job;
    setStatus(`Job ${job.id} status: ${job.status}`);
    setProgress(job.progress || 0);
    pushChartPoint(job.progress || 0);
    renderMetrics(job);
    if (job.status === "completed" || job.status === "failed") {
      if (pollTimer) {
        window.clearInterval(pollTimer);
        pollTimer = null;
      }
      activeJobId = null;
      if (job.status === "failed" && job.error_message) {
        jobMetrics.textContent = `${jobMetrics.textContent}\n\nError: ${job.error_message}`;
      }
      await Promise.all([fetchModels(), fetchReadiness()]);
      if (eventSource) {
        eventSource.close();
        eventSource = null;
      }
    }
  }

  function startJobEventStream(jobId) {
    if (!window.EventSource) {
      return false;
    }
    if (eventSource) {
      eventSource.close();
    }
    eventSource = new EventSource(`${apiBase}/training/jobs/${jobId}/events`);
    eventSource.addEventListener("update", async (event) => {
      const job = JSON.parse(event.data);
      setStatus(`Job ${job.id} status: ${job.status}`);
      setProgress(job.progress || 0);
      pushChartPoint(job.progress || 0);
      renderMetrics(job);
      if (job.status === "completed" || job.status === "failed") {
        if (job.status === "failed" && job.error_message) {
          jobMetrics.textContent = `${jobMetrics.textContent}\n\nError: ${job.error_message}`;
        }
        eventSource.close();
        eventSource = null;
        activeJobId = null;
        await Promise.all([fetchModels(), fetchReadiness()]);
      }
    });
    eventSource.addEventListener("end", () => {
      if (eventSource) {
        eventSource.close();
        eventSource = null;
      }
      activeJobId = null;
      fetchModels().catch(() => null);
      fetchReadiness().catch(() => null);
    });
    eventSource.onerror = () => {
      if (eventSource) {
        eventSource.close();
        eventSource = null;
      }
      if (!pollTimer && activeJobId) {
        pollTimer = window.setInterval(pollJobOnce, 1000);
      }
    };
    return true;
  }

  trainForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = toPayload(new FormData(trainForm));
    const response = await fetch(`${apiBase}/training/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await safeJson(response);
    if (!response.ok) {
      setStatus(`Failed to create job: ${body.error || "unknown error"}`);
      return;
    }
    activeJobId = body.job.id;
    setStatus(`Training job ${activeJobId} queued`);
    setProgress(0);
    renderMetrics({ metrics: null });
    resetChart();
    pushChartPoint(0);
    if (pollTimer) {
      window.clearInterval(pollTimer);
      pollTimer = null;
    }
    const usingSse = startJobEventStream(activeJobId);
    if (!usingSse) {
      pollTimer = window.setInterval(pollJobOnce, 1000);
      await pollJobOnce();
    }
  });

  modelTypeInput.addEventListener("change", updateMlpHint);

  [lookbackInput, selectionModeInput, actorScopeInput].forEach((input) => {
    input.addEventListener("change", () => {
      fetchReadiness().catch((err) => {
        readinessStatus.textContent = `Readiness check failed: ${String(err)}`;
      });
    });
  });

  refreshSessionsBtn.addEventListener("click", () => {
    fetchSessions().catch((err) => {
      setStatus(`Session refresh failed: ${String(err)}`);
    });
  });

  refreshModelsBtn.addEventListener("click", () => {
    fetchModels().catch((err) => {
      setStatus(`Model refresh failed: ${String(err)}`);
    });
  });

  updateMlpHint();
  resetChart();
  fetchReadiness().catch((err) => {
    readinessStatus.textContent = `Readiness check failed: ${String(err)}`;
  });
  fetchSessions().catch((err) => {
    setStatus(`Session refresh failed: ${String(err)}`);
  });
  fetchModels().catch((err) => {
    setStatus(`Model refresh failed: ${String(err)}`);
  });
})();
