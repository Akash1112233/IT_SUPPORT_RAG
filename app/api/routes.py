from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from pydantic import BaseModel, Field
from app.core.config import get_settings
from app.rag.workflow import ask
from app.rag.vectorstore import add_documents 
from app.services.ingestion import load_file, chunk_documents, SUPPORTED
from app.services.audit import write_audit
from app.services.document_identity import calculate_document_hash

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




@router.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    x_admin_key: str = Header(default=""),
):
    if not settings.admin_api_key or x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")

    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()

    if not filename or suffix not in SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Supported: {', '.join(sorted(SUPPORTED))}",
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    upload_dir = Path(settings.upload_dir).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)

    dest = upload_dir / filename

    try:
        # Hash the original bytes, not the extracted text.
        document_hash = calculate_document_hash(file_bytes)

        # Store the upload after validating its extension.
        dest.write_bytes(file_bytes)

        docs = load_file(dest)

        # Attach document identity before splitting into chunks.
        for doc in docs:
            doc.metadata.update({
                "document_name": filename,
                "document_hash": document_hash,
            })

        chunks = chunk_documents(docs)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No readable text was found in the uploaded file.",
            )

        ids = add_documents(chunks)

        return {
            "message": "Document indexed",
            "file": filename,
            "document_hash": document_hash,
            "chunks": len(chunks),
            "ids_created": len(ids),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed. Check the application logs.",
        ) from exc
    finally:
        await file.close()
