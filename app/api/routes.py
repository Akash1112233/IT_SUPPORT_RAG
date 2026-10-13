import logging
import zipfile
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.rag.workflow import ask
from app.rag.vectorstore import add_documents, delete_documents
from app.services.ingestion import load_file, chunk_documents, SUPPORTED
from app.services.audit import write_audit
from app.services.document_identity import calculate_document_hash
from app.services.document_manifest import (
    get_document,
    is_document_unchanged,
    mark_document_indexed,
    mark_document_pending,
)

router = APIRouter(prefix="/api")
settings = get_settings()

class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000)


@router.get("/health")
def health():
    return {"status": "ok", "service": settings.app_name}


@router.post("/chat")
def chat(payload: ChatRequest):
    try:
        result = ask(payload.question)
        write_audit(payload.question, result["source_used"], result.get("trace", []))

        return {
            "answer": result["answer"],
            "source_used": result["source_used"],
            "trace": result.get("trace", []),
            "citations": result.get("citations", []),
            "rewritten_query": result.get("current_query", payload.question),

        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc



def validate_uploaded_content(filename: str, content: bytes) -> None:
    """Reject obviously invalid files before document parsing."""
    suffix = Path(filename).suffix.lower()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if suffix == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not a valid PDF.",
            )

    elif suffix == ".docx":
        # DOCX files are ZIP containers. This checks their
        # container structure, not every aspect of the document.
        from io import BytesIO

        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                names = set(archive.namelist())
                if (
                    "[Content_Types].xml" not in names
                    or "word/document.xml" not in names
                ):
                    raise HTTPException(
                        status_code=400,
                        detail="The uploaded file is not a valid DOCX.",
                    )
                if archive.testzip() is not None:
                    raise HTTPException(
                        status_code=400,
                        detail="The DOCX archive is corrupted.",
                    )
        except zipfile.BadZipFile as exc:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not a valid DOCX.",
            ) from exc

    elif suffix in {".txt", ".md"}:
        try:
            content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail="Text files must use UTF-8 encoding.",
            ) from exc


@router.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    x_admin_key: str = Header(default=""),
):
    # Require a configured admin key.
    if not settings.admin_api_key or x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")

    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()

    if not filename or suffix not in SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Supported: {', '.join(sorted(SUPPORTED))}",
        )

    try:
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        # Step 1: Identify the exact uploaded file version.
        document_hash = calculate_document_hash(file_bytes)

        # Step 2: Save the file for the existing document loaders.
        upload_dir = Path(settings.upload_dir).resolve()
        upload_dir.mkdir(parents=True, exist_ok=True)

        destination = upload_dir / filename
        destination.write_bytes(file_bytes)

        # Step 3: Skip files whose current version is already indexed.
        if is_document_unchanged(filename, document_hash):
            return {
                "message": "Document is unchanged; indexing skipped.",
                "file": filename,
                "document_hash": document_hash,
                "chunks": 0,
                "skipped": True,
            }

        # Keep the previous manifest entry so we know which old
        # vectors belong to this document.
        previous_record = get_document(filename)
        old_ids = (
            previous_record["chunk_ids"]
            if previous_record is not None
            else []
        )

        # Step 4: Mark this version pending. The manifest retains
        # the previous chunk IDs until indexing completes.
        mark_document_pending(filename, document_hash)

        # Step 5: Load and split the document.
        documents = load_file(destination)

        for document in documents:
            document.metadata.update({
                "document_name": filename,
                "document_hash": document_hash,
            })

        chunks = chunk_documents(documents)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No readable text was found in the document.",
            )

        # Step 6: Upsert the new version using deterministic IDs.
        new_ids = add_documents(chunks)

        if not new_ids:
            raise RuntimeError(
                "Vector store did not return IDs for the indexed chunks."
            )

        # Step 7: Delete only obsolete IDs from the previous
        # manifest version. New vectors are already indexed.
        new_id_set = set(new_ids)
        obsolete_ids = [
            old_id for old_id in old_ids
            if old_id not in new_id_set
        ]

        delete_documents(obsolete_ids)

        # Step 8: Commit the new version to the manifest only
        # after indexing and obsolete-vector deletion succeed.
        mark_document_indexed(
            document_name=filename,
            document_hash=document_hash,
            chunk_ids=new_ids,
        )

        return {
            "message": "Document indexed successfully.",
            "file": filename,
            "document_hash": document_hash,
            "chunks": len(new_ids),
            "skipped": False,
            "obsolete_chunks_removed": len(obsolete_ids),
        }

    except HTTPException:
        raise

    except Exception as exc:
        # Log the exception server-side rather than exposing
        # internal implementation details to API clients.
        import logging
        logging.getLogger(__name__).exception(
            "Document ingestion failed for %s",
            filename,
        )

        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed. Check application logs.",
        ) from exc

    finally:
        await file.close()
