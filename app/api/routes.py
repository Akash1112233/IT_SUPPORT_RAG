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


import logging
from pathlib import Path
from fastapi import APIRouter, File, Header, HTTPException, UploadFile

# Assuming router, settings, SUPPORTED, calculate_document_hash, is_document_unchanged,
# get_document, mark_document_pending, load_file, chunk_documents, add_documents,
# delete_documents, mark_document_indexed, and validate_uploaded_content are imported/defined.




@router.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    x_admin_key: str = Header(default=""),
):
    filename = Path(file.filename or "").name
    temporary_path = None

    try:
        # 1. Authenticate the upload request.
        if (
            not settings.admin_api_key
            or x_admin_key != settings.admin_api_key
        ):
            raise HTTPException(
                status_code=401,
                detail="Invalid admin key",
            )

        # 2. Validate the filename and extension.
        suffix = Path(filename).suffix.lower()

        if not filename or suffix not in SUPPORTED:
            raise HTTPException(
                status_code=400,
                detail=f"Supported: {', '.join(sorted(SUPPORTED))}",
            )

        # 3. Read the file with a strict size limit.
        max_bytes = settings.max_upload_size_mb * 1024 * 1024
        file_bytes = bytearray()

        while True:
            remaining = max_bytes + 1 - len(file_bytes)

            if remaining <= 0:
                break

            block = await file.read(min(64 * 1024, remaining))

            if not block:
                break

            file_bytes.extend(block)

            if len(file_bytes) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "File exceeds the maximum upload size of "
                        f"{settings.max_upload_size_mb} MB."
                    ),
                )

        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        content = bytes(file_bytes)

        # 4. Validate content before saving or parsing it.
        validate_uploaded_content(filename, content)

        # 5. Identify this exact document version.
        document_hash = calculate_document_hash(content)

        # 6. Skip indexing if this version is already indexed.
        if is_document_unchanged(filename, document_hash):
            return {
                "message": "Document is unchanged; indexing skipped.",
                "file": filename,
                "document_hash": document_hash,
                "chunks": 0,
                "skipped": True,
            }

        # 7. Read the previous manifest before marking this version
        # pending. Preserve the IDs needed for safe replacement.
        previous_record = get_document(filename)

        old_ids = (
            previous_record["chunk_ids"]
            if previous_record is not None
            else []
        )

        # 8. Create a separate file for this content version.
        upload_dir = Path(settings.upload_dir).resolve()
        upload_dir.mkdir(parents=True, exist_ok=True)

        original_path = Path(filename)
        versioned_filename = (
            f"{original_path.stem}_{document_hash[:12]}"
            f"{original_path.suffix.lower()}"
        )
        destination = upload_dir / versioned_filename

        # Do not rewrite a versioned file that already exists.
        if not destination.exists():
            with NamedTemporaryFile(
                mode="wb",
                dir=upload_dir,
                suffix=suffix,
                delete=False,
            ) as temp_file:
                temporary_path = Path(temp_file.name)
                temp_file.write(content)

            # Atomically publish the completed temporary file.
            temporary_path.replace(destination)
            temporary_path = None

        # 9. Record that this version is being processed.
        # Existing chunk IDs remain in the manifest on conflict.
        mark_document_pending(filename, document_hash)

        # 10. Load and split the versioned file.
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

        # 11. Index the new version with deterministic vector IDs.
        new_ids = add_documents(chunks)

        if not new_ids or len(new_ids) != len(chunks):
            raise RuntimeError(
                "The vector store did not return an ID for every chunk."
            )

        # 12. Remove only obsolete IDs recorded for this document.
        new_id_set = set(new_ids)
        obsolete_ids = [
            old_id for old_id in old_ids
            if old_id not in new_id_set
        ]

        delete_documents(obsolete_ids)

        # 13. Mark the new version indexed only after the preceding
        # vector operations have completed successfully.
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
        logging.getLogger(__name__).exception(
            "Document ingestion failed for %s",
            filename or "<unnamed file>",
        )

        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed. Check application logs.",
        ) from exc

    finally:
        # Remove an incomplete temporary file if writing failed.
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                logging.getLogger(__name__).warning(
                    "Could not remove temporary upload file."
                )

        await file.close()


