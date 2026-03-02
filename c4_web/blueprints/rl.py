"""RL job API routes and event streams."""

from __future__ import annotations

import json
import time

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context

rl_bp = Blueprint("rl_api", __name__, url_prefix="/api/v1/rl")


def _repo():
    return current_app.extensions["repository"]


def _jobs():
    return current_app.extensions["rl_jobs"]


def _decode(raw):
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _serialize(job: dict) -> dict:
    return {
        "id": int(job["id"]),
        "status": job["status"],
        "params": _decode(job["params_json"]),
        "progress": float(job["progress"]),
        "metrics": _decode(job["metrics_json"]),
        "error_message": job["error_message"],
        "model_id": int(job["model_id"]) if job["model_id"] is not None else None,
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }


@rl_bp.get("/status")
def rl_status():
    return jsonify(
        {
            "status": "enabled",
            "message": "RL self-play job API is available.",
        }
    )


@rl_bp.post("/jobs")
def create_rl_job():
    payload = request.get_json(silent=True) or {}
    payload.setdefault("episodes", 300)
    payload.setdefault("alpha", 0.1)
    payload.setdefault("gamma", 0.6)
    payload.setdefault("epsilon_start", 0.99)
    payload.setdefault("epsilon_end", 0.1)
    payload.setdefault("epsilon_decay", 0.9999)
    payload.setdefault("alpha_decay_step", 1000)
    payload.setdefault("alpha_decay_rate", 0.9)
    payload.setdefault("opponent", "negamax")
    payload.setdefault("switch_prob", 0.5)
    payload.setdefault("seed", 7)
    job = _jobs().submit_job(payload)
    return jsonify({"job": _serialize(job)}), 202


@rl_bp.get("/jobs")
def list_rl_jobs():
    jobs = _repo().list_rl_jobs(limit=100)
    return jsonify({"jobs": [_serialize(job) for job in jobs]})


@rl_bp.get("/jobs/<int:job_id>")
def get_rl_job(job_id: int):
    job = _repo().get_rl_job(job_id)
    if job is None:
        return jsonify({"error": "RL job not found."}), 404
    return jsonify({"job": _serialize(job)})


@rl_bp.get("/jobs/<int:job_id>/events")
def stream_rl_job_events(job_id: int):
    if _repo().get_rl_job(job_id) is None:
        return jsonify({"error": "RL job not found."}), 404

    @stream_with_context
    def event_stream():
        last_payload = None
        while True:
            job = _repo().get_rl_job(job_id)
            if job is None:
                yield "event: error\ndata: {\"error\": \"RL job not found\"}\n\n"
                break
            payload = json.dumps(_serialize(job), sort_keys=True)
            if payload != last_payload:
                yield f"event: update\ndata: {payload}\n\n"
                last_payload = payload
            if job["status"] in {"completed", "failed"}:
                yield "event: end\ndata: {}\n\n"
                break
            time.sleep(0.5)

    return Response(event_stream(), mimetype="text/event-stream")
