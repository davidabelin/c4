"""Storage helpers and repository facade for c4."""

from c4_storage.object_store import is_gcs_uri, join_storage_path, read_bytes, write_bytes, write_text
from c4_storage.repository import C4Repository

__all__ = [
    "C4Repository",
    "is_gcs_uri",
    "join_storage_path",
    "write_bytes",
    "read_bytes",
    "write_text",
]
