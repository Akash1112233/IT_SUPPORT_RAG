
import hashlib
import json
from typing import Any


def calculate_document_hash(content: bytes) -> str:
    """
    Generate a SHA-256 fingerprint for the original document bytes.

    Identical file contents produce the same hash.
    Different file contents normally produce different hashes.
    """
    return hashlib.sha256(content).hexdigest()


def calculate_chunk_id(
    document_hash: str,
    chunk_index: int,
    chunk_text: str,
) -> str:
    """
    Generate a deterministic ID for a document chunk.

    The same document version, chunk position, and chunk text
    produce the same ID when processed again.
    """
    identity = {
        "document_hash": document_hash,
        "chunk_index": chunk_index,
        "chunk_text": chunk_text,
    }

    serialized_identity = json.dumps(
        identity,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        serialized_identity.encode("utf-8")
    ).hexdigest()


def build_chunk_metadata(
    document_name: str,
    document_hash: str,
    chunk_index: int,
    chunk_text: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build metadata that can later be stored alongside a vector.
    """
    metadata = {
        "document_name": document_name,
        "document_hash": document_hash,
        "chunk_index": chunk_index,
        "chunk_hash": hashlib.sha256(
            chunk_text.encode("utf-8")
        ).hexdigest(),
    }

    if extra_metadata:
        # Prevent caller-supplied values from overwriting
        # the identity fields defined above.
        reserved_keys = set(metadata)
        for key, value in extra_metadata.items():
            if key not in reserved_keys:
                metadata[key] = value

    return metadata
