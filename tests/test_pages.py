from __future__ import annotations

from pathlib import Path

import pytest

from c4_web import create_app


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(
        {
            "TESTING": True,
            "DB_PATH": str(tmp_path / "c4.db"),
            "EVENTS_DIR": str(tmp_path / "events"),
            "MODELS_DIR": str(tmp_path / "models"),
            "EXPORTS_DIR": str(tmp_path / "exports"),
            "DRL_HOME_URL": "https://drl.example.test/",
        }
    )
    return app.test_client()


@pytest.mark.parametrize("path", ["/", "/play", "/arena", "/training", "/rl"])
def test_pages_are_standalone(client, path: str):
    response = client.get(path)
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "copyleft.svg" in html
    assert 'width="16"' in html
    assert 'height="16"' in html
    assert "2026 Connect4 Lab" in html
    assert "DRL Labs" in html
    assert "https://drl.example.test/" in html
    forbidden_fragments = (
        "Back to " + "AI" + "X Hub",
        "__" + "AI" + "X_HUB_URL__",
        "AI" + "X " + "Proto" + "dyne",
    )
    for fragment in forbidden_fragments:
        assert fragment not in html
