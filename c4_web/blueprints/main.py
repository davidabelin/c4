"""Page-rendering and health routes for server-rendered c4 web UI."""

from __future__ import annotations

from datetime import UTC, datetime

from flask import Blueprint, current_app, jsonify, render_template

main_bp = Blueprint("main", __name__)


@main_bp.get("/")
def home() -> str:
    """Render the c4 landing page."""

    return render_template("pages/home.html")


@main_bp.get("/play")
def play_page() -> str:
    """Render the interactive single-game page."""

    return render_template("pages/play.html", default_agent=current_app.config["DEFAULT_AGENT"])


@main_bp.get("/arena")
def arena_page() -> str:
    """Render the arena match dashboard."""

    return render_template("pages/arena.html", default_agent=current_app.config["DEFAULT_AGENT"])


@main_bp.get("/training")
def training_page() -> str:
    """Render the supervised-training page."""

    return render_template("pages/train.html")


@main_bp.get("/rl")
def rl_page() -> str:
    """Render the RL training page."""

    return render_template("pages/rl.html")


@main_bp.get("/healthz")
def healthz():
    """Return a lightweight health payload for uptime checks."""

    return jsonify(
        {
            "status": "ok",
            "service": "c4-web",
            "timestamp": datetime.now(UTC).isoformat(),
        }
    )
