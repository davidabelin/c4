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
            "AIX_HUB_URL": "https://aix-labs.uw.r.appspot.com/",
        }
    )
    return app.test_client()


@pytest.mark.parametrize("path", ["/", "/play", "/arena", "/training", "/rl"])
def test_pages_include_aix_footer(client, path: str):
    response = client.get(path)
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "GNU copyright 2026 AIX Protodyne" in html
    assert "Contact Us" in html
    assert "Privacy" in html
    assert "AIX TOC" in html
