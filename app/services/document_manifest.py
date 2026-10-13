
import json
import sqlite3
from pathlib import Path
from typing import Any

from app.core.config import get_settings


settings = get_settings()

# Keep the manifest separate from the audit database.
MANIFEST_DB = (
    Path(settings.audit_db_path).resolve().parent
    / "document_manifest.db"
)


def get_connection():
    """Open a SQLite connection and ensure its folder exists."""
    MANIFEST_DB.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        str(MANIFEST_DB),
        timeout=10,
    )
    connection.row_factory = sqlite3.Row
    return connection


def init_manifest():
    """Create the document manifest table if it doesn't exist."""
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS document_manifest (
                document_name TEXT PRIMARY KEY,
                document_hash TEXT NOT NULL,
                chunk_ids TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT
                    (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
        """)
        connection.commit()


def get_document(document_name: str) -> dict[str, Any] | None:
    """Return the latest recorded version of a document."""
    init_manifest()

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT document_name, document_hash, chunk_ids,
                   chunk_count, status, updated_at
            FROM document_manifest
            WHERE document_name = ?
            """,
            (document_name,),
        ).fetchone()

    if row is None:
        return None

    result = dict(row)
    result["chunk_ids"] = json.loads(result["chunk_ids"])
    return result


def is_document_unchanged(
    document_name: str,
    document_hash: str,
) -> bool:
    """Return True only for a previously indexed matching version."""
    record = get_document(document_name)

    return (
        record is not None
        and record["status"] == "indexed"
        and record["document_hash"] == document_hash
    )


def mark_document_indexed(
    document_name: str,
    document_hash: str,
    chunk_ids: list[str],
):
    """
    Record a successfully indexed document version.

    Call this only after the vector-store operation succeeds.
    """
    init_manifest()

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO document_manifest (
                document_name,
                document_hash,
                chunk_ids,
                chunk_count,
                status,
                updated_at
            )
            VALUES (?, ?, ?, ?, 'indexed',
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            ON CONFLICT(document_name) DO UPDATE SET
                document_hash = excluded.document_hash,
                chunk_ids = excluded.chunk_ids,
                chunk_count = excluded.chunk_count,
                status = 'indexed',
                updated_at = excluded.updated_at
            """,
            (
                document_name,
                document_hash,
                json.dumps(chunk_ids),
                len(chunk_ids),
            ),
        )
        connection.commit()


def mark_document_pending(
    document_name: str,
    document_hash: str,
):
    """
    Record that a document version is awaiting indexing.

    This does not delete or modify existing Pinecone vectors.
    """
    init_manifest()

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO document_manifest (
                document_name,
                document_hash,
                chunk_ids,
                chunk_count,
                status,
                updated_at
            )
            VALUES (?, ?, '[]', 0, 'pending',
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            ON CONFLICT(document_name) DO UPDATE SET
                document_hash = excluded.document_hash,
                status = 'pending',
                updated_at = excluded.updated_at
            """,
            (document_name, document_hash),
        )
        connection.commit()
