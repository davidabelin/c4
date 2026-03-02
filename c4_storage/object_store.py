"""Storage abstraction for local filesystem and Google Cloud Storage objects."""

from __future__ import annotations

from pathlib import Path


def is_gcs_uri(path: str) -> bool:
    """Return ``True`` when path uses ``gs://`` scheme."""

    return str(path).startswith("gs://")


def join_storage_path(root: str, *parts: str) -> str:
    """Join path components for local paths or ``gs://`` URIs."""

    if is_gcs_uri(root):
        tokens = [str(root).rstrip("/")]
        for part in parts:
            clean = str(part).strip("/")
            if clean:
                tokens.append(clean)
        return "/".join(tokens)
    path = Path(root)
    for part in parts:
        path = path / str(part)
    return str(path)


def _split_gcs_uri(uri: str) -> tuple[str, str]:
    """Split ``gs://bucket/blob`` URI into bucket and object path."""

    raw = str(uri)
    if not raw.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    remainder = raw[len("gs://") :]
    if "/" not in remainder:
        return remainder, ""
    bucket, blob = remainder.split("/", 1)
    return bucket, blob


def _get_storage_client():
    """Create a Google Cloud Storage client lazily."""

    try:
        from google.cloud import storage
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("google-cloud-storage is required for gs:// paths") from exc
    return storage.Client()


def write_bytes(destination: str, payload: bytes, *, content_type: str = "application/octet-stream") -> str:
    """Write bytes to local path or GCS URI."""

    if is_gcs_uri(destination):
        bucket_name, blob_name = _split_gcs_uri(destination)
        if not blob_name:
            raise ValueError(f"GCS destination must include object path: {destination}")
        client = _get_storage_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(payload, content_type=content_type)
        return destination
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return str(path)


def read_bytes(source: str) -> bytes:
    """Read bytes from local path or GCS URI."""

    if is_gcs_uri(source):
        bucket_name, blob_name = _split_gcs_uri(source)
        if not blob_name:
            raise ValueError(f"GCS source must include object path: {source}")
        client = _get_storage_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.download_as_bytes()
    return Path(source).read_bytes()


def write_text(destination: str, text: str, *, content_type: str = "text/plain; charset=utf-8") -> str:
    """Write UTF-8 text to local path or GCS URI."""

    return write_bytes(destination, text.encode("utf-8"), content_type=content_type)
