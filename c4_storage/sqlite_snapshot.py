"""Best-effort SQLite snapshot mirroring for low-cost cloud persistence."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Protocol


class _Blob(Protocol):
    def exists(self) -> bool: ...
    def download_to_filename(self, filename: str) -> None: ...
    def upload_from_filename(self, filename: str, *, content_type: str) -> None: ...


class _Bucket(Protocol):
    def blob(self, blob_name: str) -> _Blob: ...


class _StorageClient(Protocol):
    def bucket(self, bucket_name: str) -> _Bucket: ...


def _split_gcs_uri(uri: str) -> tuple[str, str]:
    raw = str(uri).strip()
    if not raw.startswith("gs://"):
        raise ValueError(f"Snapshot URI must be a gs:// object path: {uri}")
    remainder = raw[len("gs://") :]
    if "/" not in remainder:
        raise ValueError(f"Snapshot URI must include a bucket and object path: {uri}")
    bucket, blob = remainder.split("/", 1)
    if not bucket or not blob:
        raise ValueError(f"Snapshot URI must include a bucket and object path: {uri}")
    return bucket, blob


def _default_client() -> _StorageClient:
    from google.cloud import storage

    return storage.Client()


class SQLiteSnapshotMirror:
    """Mirror one local SQLite file to a Cloud Storage object when configured."""

    def __init__(
        self,
        *,
        db_path: str,
        snapshot_uri: str,
        logger,
        client_factory: Callable[[], _StorageClient] = _default_client,
    ) -> None:
        self.db_path = Path(db_path)
        self.snapshot_uri = str(snapshot_uri or "").strip()
        self.logger = logger
        self._client_factory = client_factory
        self._last_signature: tuple[int, int, str] | None = None
        self._remote_existed = False

    @property
    def enabled(self) -> bool:
        return bool(self.snapshot_uri)

    def _blob(self) -> _Blob:
        bucket_name, blob_name = _split_gcs_uri(self.snapshot_uri)
        return self._client_factory().bucket(bucket_name).blob(blob_name)

    def _signature(self) -> tuple[int, int, str] | None:
        if not self.db_path.exists():
            return None
        stat = self.db_path.stat()
        digest = hashlib.sha256(self.db_path.read_bytes()).hexdigest()
        return (stat.st_size, stat.st_mtime_ns, digest)

    def download_if_missing(self) -> None:
        if not self.enabled or self.db_path.exists():
            return
        try:
            blob = self._blob()
            self._remote_existed = bool(blob.exists())
            if not self._remote_existed:
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(self.db_path))
            self._last_signature = self._signature()
        except Exception as exc:
            self.logger.warning("SQLite snapshot download skipped for %s: %s", self.snapshot_uri, exc)

    def upload_if_changed(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        signature = self._signature()
        if signature is None:
            return
        if not force and signature == self._last_signature:
            return
        try:
            self._blob().upload_from_filename(str(self.db_path), content_type="application/vnd.sqlite3")
            self._last_signature = signature
            self._remote_existed = True
        except Exception as exc:
            self.logger.warning("SQLite snapshot upload skipped for %s: %s", self.snapshot_uri, exc)

    def sync_after_schema_init(self) -> None:
        self.upload_if_changed(force=not self._remote_existed)
