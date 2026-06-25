from __future__ import annotations

from pathlib import Path

from c4_storage.sqlite_snapshot import SQLiteSnapshotMirror


class _Logger:
    def warning(self, message: str, *args) -> None:
        pass


class _Blob:
    def __init__(self, *, payload: bytes | None = None) -> None:
        self.payload = payload
        self.uploads: list[bytes] = []

    def exists(self) -> bool:
        return self.payload is not None

    def download_to_filename(self, filename: str) -> None:
        Path(filename).write_bytes(self.payload or b"")

    def upload_from_filename(self, filename: str, *, content_type: str) -> None:
        self.uploads.append(Path(filename).read_bytes())


class _Client:
    def __init__(self, blob: _Blob) -> None:
        self._blob = blob

    def bucket(self, bucket_name: str):
        return self

    def blob(self, blob_name: str):
        return self._blob


def test_snapshot_downloads_existing_gcs_object(tmp_path: Path):
    blob = _Blob(payload=b"sqlite bytes")
    mirror = SQLiteSnapshotMirror(
        db_path=str(tmp_path / "db.sqlite3"),
        snapshot_uri="gs://bucket/path/db.sqlite3",
        logger=_Logger(),
        client_factory=lambda: _Client(blob),
    )

    mirror.download_if_missing()

    assert (tmp_path / "db.sqlite3").read_bytes() == b"sqlite bytes"


def test_snapshot_noops_without_uri(tmp_path: Path):
    mirror = SQLiteSnapshotMirror(
        db_path=str(tmp_path / "missing.sqlite3"),
        snapshot_uri="",
        logger=_Logger(),
    )

    mirror.download_if_missing()
    mirror.upload_if_changed()

    assert not (tmp_path / "missing.sqlite3").exists()


def test_snapshot_uploads_only_when_file_changes(tmp_path: Path):
    blob = _Blob()
    db_path = tmp_path / "db.sqlite3"
    db_path.write_bytes(b"one")
    mirror = SQLiteSnapshotMirror(
        db_path=str(db_path),
        snapshot_uri="gs://bucket/path/db.sqlite3",
        logger=_Logger(),
        client_factory=lambda: _Client(blob),
    )

    mirror.upload_if_changed()
    mirror.upload_if_changed()
    db_path.write_bytes(b"two")
    mirror.upload_if_changed()

    assert blob.uploads == [b"one", b"two"]
