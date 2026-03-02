"""Training and model-registry API routes."""

from __future__ import annotations

import json
import time

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context

from c4_training.supervised import training_readiness


training_bp = Blueprint("training_api", __name__, url_prefix="/api/v1")


def _repo():
    return current_app.extensions["repository"]


def _jobs():
    return current_app.extensions["training_jobs"]


def _decode_json(raw_value):
    if raw_value is None:
        return None
    try:
        return json.loads(raw_value)
    except Exception:
        return raw_value


@training_bp.post("/training/jobs")
def create_training_job():
    payload = request.get_json(silent=True) or {}
    payload.setdefault("model_type", "decision_tree")
    payload.setdefault("lookback", 5)
    payload.setdefault("test_size", 0.2)
    payload.setdefault("learning_rate", 0.001)
    payload.setdefault("hidden_layer_sizes", [64, 32])
    payload.setdefault("epochs", 200)
    payload.setdefault("batch_size", "auto")
    payload.setdefault("random_state", 42)
    job = _jobs().submit_job(payload)
    return jsonify({"job": _serialize_job(job)}), 202


@training_bp.post("/internal/training/jobs/<int:job_id>/run")
def run_training_job_internal(job_id: int):
    expected = current_app.config.get("INTERNAL_WORKER_TOKEN") or current_app.config.get("TRAINING_WORKER_TOKEN")
    provided = request.headers.get("X-Worker-Token", "")
    if expected and provided != expected:
        return jsonify({"error": "Unauthorized worker token."}), 403
    try:
        _jobs().run_job_by_id(job_id)
    except KeyError:
        return jsonify({"error": "Training job not found."}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    job = _repo().get_training_job(job_id)
    if job is None:
        return jsonify({"error": "Training job not found."}), 404
    return jsonify({"job": _serialize_job(job)})


@training_bp.get("/training/readiness")
def get_training_readiness():
    lookback = int(request.args.get("lookback", 5))
    rows = _repo().list_ai_moves_for_training()
    summary = training_readiness(rows=rows, lookback=lookback, minimum_samples=20)
    return jsonify({"readiness": summary})


@training_bp.get("/training/jobs")
def list_training_jobs():
    jobs = _repo().list_training_jobs(limit=100)
    return jsonify({"jobs": [_serialize_job(job) for job in jobs]})


@training_bp.get("/training/jobs/<int:job_id>")
def get_training_job(job_id: int):
    job = _repo().get_training_job(job_id)
    if job is None:
        return jsonify({"error": "Training job not found."}), 404
    return jsonify({"job": _serialize_job(job)})


@training_bp.get("/training/jobs/<int:job_id>/events")
def stream_training_job_events(job_id: int):
    if _repo().get_training_job(job_id) is None:
        return jsonify({"error": "Training job not found."}), 404

    @stream_with_context
    def event_stream():
        last_payload = None
        while True:
            job = _repo().get_training_job(job_id)
            if job is None:
                yield "event: error\ndata: {\"error\": \"Training job not found\"}\n\n"
                break
            payload = json.dumps(_serialize_job(job), sort_keys=True)
            if payload != last_payload:
                yield f"event: update\ndata: {payload}\n\n"
                last_payload = payload
            status = job["status"]
            if status in {"completed", "failed"}:
                yield "event: end\ndata: {}\n\n"
                break
            time.sleep(0.5)

    return Response(event_stream(), mimetype="text/event-stream")


@training_bp.get("/models")
def list_models():
    models = _repo().list_models(limit=200)
    return jsonify({"models": [_serialize_model(model) for model in models]})


@training_bp.post("/models/<int:model_id>/activate")
def activate_model(model_id: int):
    model = _repo().activate_model(model_id)
    if model is None:
        return jsonify({"error": "Model not found."}), 404
    return jsonify({"model": _serialize_model(model)})


def _serialize_job(job: dict) -> dict:
    return {
        "id": int(job["id"]),
        "status": job["status"],
        "model_type": job["model_type"],
        "params": _decode_json(job["params_json"]),
        "progress": float(job["progress"]),
        "metrics": _decode_json(job["metrics_json"]),
        "error_message": job["error_message"],
        "model_id": int(job["model_id"]) if job["model_id"] is not None else None,
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }


def _serialize_model(model: dict) -> dict:
    return {
        "id": int(model["id"]),
        "name": model["name"],
        "model_type": model["model_type"],
        "artifact_path": model["artifact_path"],
        "lookback": int(model["lookback"]),
        "metrics": _decode_json(model["metrics_json"]),
        "is_active": bool(model["is_active"]),
        "created_at": model["created_at"],
    }
