(function () {
  "use strict";

  const appBasePath = String(window.__APP_BASE_PATH__ || "").replace(/\/+$/, "");
  const apiBase = `${appBasePath}/api/v1`;

  const rlForm = document.getElementById("rlForm");
  const rlJobStatus = document.getElementById("rlJobStatus");
  const rlJobProgress = document.getElementById("rlJobProgress");
  const rlJobMetrics = document.getElementById("rlJobMetrics");
  const refreshRlJobsBtn = document.getElementById("refreshRlJobsBtn");
  const rlJobsTableBody = document.getElementById("rlJobsTableBody");

  let activeRlJobId = null;
  let rlPollTimer = null;
  let rlEventSource = null;

  function setStatus(text) {
    rlJobStatus.textContent = text;
  }

  function setProgress(progress) {
    const pct = Math.max(0, Math.min(100, Math.round((progress || 0) * 100)));
    rlJobProgress.style.width = `${pct}%`;
  }

  function payloadFromForm(formData) {
    return {
      episodes: Number(formData.get("episodes") || 300),
      opponent: String(formData.get("opponent") || "alpha_beta_v9"),
      alpha: Number(formData.get("alpha") || 0.1),
      gamma: Number(formData.get("gamma") || 0.6),
      epsilon_start: Number(formData.get("epsilon_start") || 0.99),
      epsilon_end: Number(formData.get("epsilon_end") || 0.1),
      epsilon_decay: Number(formData.get("epsilon_decay") || 0.9999),
      alpha_decay_step: Number(formData.get("alpha_decay_step") || 1000),
      alpha_decay_rate: Number(formData.get("alpha_decay_rate") || 0.9),
      switch_prob: Number(formData.get("switch_prob") || 0.5),
      seed: Number(formData.get("seed") || 7),
    };
  }

  async function fetchRlJobs() {
    const response = await fetch(`${apiBase}/rl/jobs`);
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.error || "Failed to load RL jobs");
    }
    rlJobsTableBody.innerHTML = "";
    body.jobs.forEach((job) => {
      const tr = document.createElement("tr");
      const avgReward = job.metrics && job.metrics.final_100_avg_reward !== undefined
        ? Number(job.metrics.final_100_avg_reward).toFixed(3)
        : "-";
      const opponent = job.params && job.params.opponent ? String(job.params.opponent) : "-";
      tr.innerHTML = `
        <td>${job.id}</td>
        <td>${job.status}</td>
        <td>${Math.round((job.progress || 0) * 100)}%</td>
        <td>${job.model_id || "-"}</td>
        <td>${opponent}</td>
        <td>${avgReward}</td>
      `;
      rlJobsTableBody.appendChild(tr);
    });
  }

  function renderJob(job) {
    setStatus(`RL job ${job.id}: ${job.status}`);
    setProgress(job.progress);
    rlJobMetrics.textContent = JSON.stringify(job.metrics || {}, null, 2);
    if (job.status === "failed" && job.error_message) {
      rlJobMetrics.textContent = `${rlJobMetrics.textContent}\n\nError: ${job.error_message}`;
    }
  }

  async function pollRlJobOnce() {
    if (!activeRlJobId) {
      return;
    }
    const response = await fetch(`${apiBase}/rl/jobs/${activeRlJobId}`);
    const body = await response.json();
    if (!response.ok) {
      setStatus(`RL poll failed: ${body.error || "unknown error"}`);
      return;
    }
    const job = body.job;
    renderJob(job);
    if (job.status === "completed" || job.status === "failed") {
      activeRlJobId = null;
      if (rlPollTimer) {
        window.clearInterval(rlPollTimer);
        rlPollTimer = null;
      }
      if (rlEventSource) {
        rlEventSource.close();
        rlEventSource = null;
      }
      await fetchRlJobs();
    }
  }

  function startRlEvents(jobId) {
    if (!window.EventSource) {
      return false;
    }
    if (rlEventSource) {
      rlEventSource.close();
    }
    rlEventSource = new EventSource(`${apiBase}/rl/jobs/${jobId}/events`);
    rlEventSource.addEventListener("update", async (event) => {
      const job = JSON.parse(event.data);
      renderJob(job);
      if (job.status === "completed" || job.status === "failed") {
        if (rlEventSource) {
          rlEventSource.close();
          rlEventSource = null;
        }
        activeRlJobId = null;
        await fetchRlJobs();
      }
    });
    rlEventSource.addEventListener("end", () => {
      if (rlEventSource) {
        rlEventSource.close();
        rlEventSource = null;
      }
      activeRlJobId = null;
      fetchRlJobs().catch(() => null);
    });
    rlEventSource.onerror = () => {
      if (rlEventSource) {
        rlEventSource.close();
        rlEventSource = null;
      }
      if (!rlPollTimer && activeRlJobId) {
        rlPollTimer = window.setInterval(pollRlJobOnce, 1000);
      }
    };
    return true;
  }

  rlForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = payloadFromForm(new FormData(rlForm));
    const response = await fetch(`${apiBase}/rl/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) {
      setStatus(`Failed to create RL job: ${body.error || "unknown error"}`);
      return;
    }
    activeRlJobId = body.job.id;
    rlJobMetrics.textContent = "Training RL policy...";
    setStatus(`RL job ${activeRlJobId} queued`);
    setProgress(0);
    if (rlPollTimer) {
      window.clearInterval(rlPollTimer);
      rlPollTimer = null;
    }
    const usingSse = startRlEvents(activeRlJobId);
    if (!usingSse) {
      rlPollTimer = window.setInterval(pollRlJobOnce, 1000);
      await pollRlJobOnce();
    }
  });

  refreshRlJobsBtn.addEventListener("click", () => {
    fetchRlJobs().catch((err) => setStatus(`Failed to refresh RL jobs: ${String(err)}`));
  });

  fetchRlJobs().catch((err) => setStatus(`Failed to initialize RL page: ${String(err)}`));
})();
