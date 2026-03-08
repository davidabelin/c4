"""Arena match API routes and event streams."""

from __future__ import annotations

import json
import time

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context

arena_bp = Blueprint("arena_api", __name__, url_prefix="/api/v1/arena")


def _repo():
    return current_app.extensions["repository"]


def _jobs():
    return current_app.extensions["match_jobs"]


def _decode(raw):
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _serialize(match: dict, *, include_trace: bool = True) -> dict:
    return {
        "id": int(match["id"]),
        "status": match["status"],
        "agent_a": match["agent_a_name"],
        "agent_b": match["agent_b_name"],
        "params": _decode(match["params_json"]),
        "progress": float(match["progress"]),
        "winner": match["winner"],
        "summary": _decode(match["summary_json"]),
        "trace": (_decode(match["trace_json"]) if include_trace else None),
        "error_message": match["error_message"],
        "created_at": match["created_at"],
        "updated_at": match["updated_at"],
    }


@arena_bp.post("/matches")
def create_arena_match():
    payload = request.get_json(silent=True) or {}
    try:
        match = _jobs().submit_job(payload)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"match": _serialize(match)}), 202


@arena_bp.get("/matches")
def list_arena_matches():
    matches = _repo().list_arena_matches(limit=100)
    return jsonify({"matches": [_serialize(match, include_trace=False) for match in matches]})


@arena_bp.get("/matches/<int:match_id>")
def get_arena_match(match_id: int):
    match = _repo().get_arena_match(match_id)
    if match is None:
        return jsonify({"error": "Arena match not found."}), 404
    return jsonify({"match": _serialize(match)})


@arena_bp.get("/matches/<int:match_id>/events")
def stream_arena_match_events(match_id: int):
    if _repo().get_arena_match(match_id) is None:
        return jsonify({"error": "Arena match not found."}), 404

    @stream_with_context
    def event_stream():
        last_payload = None
        while True:
            match = _repo().get_arena_match(match_id)
            if match is None:
                yield "event: error\ndata: {\"error\": \"Arena match not found\"}\n\n"
                break
            payload = json.dumps(_serialize(match), sort_keys=True)
            if payload != last_payload:
                yield f"event: update\ndata: {payload}\n\n"
                last_payload = payload
            if match["status"] in {"completed", "failed"}:
                yield "event: end\ndata: {}\n\n"
                break
            time.sleep(0.35)

    return Response(event_stream(), mimetype="text/event-stream")
