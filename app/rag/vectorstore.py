import time
from pinecone import Pinecone , ServerlessSpec
# from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.core.config import get_settings
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

from app.services.document_identity import (
    calculate_chunk_id,
    build_chunk_metadata,
)

settings = get_settings()

_embeddings = None
_vectorstore = None

EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}

def get_embedding_dimension(model_name: str | None = None) -> int:
    name = (model_name or settings.embedding_model or "").strip()
    if not name:
        raise RuntimeError("Embedding model is not configured")

    normalized = name.lower()
    if normalized in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[normalized]
    if "text-embedding-3-small" in normalized:
        return 1536
    if "text-embedding-3-large" in normalized:
        return 3072
    if "text-embedding-ada-002" in normalized:
        return 1536
    if "all-minilm" in normalized:
        return 384

    raise ValueError(
        f"Unsupported embedding model '{model_name or settings.embedding_model}' for Pinecone. "
        "Add the matching dimension to EMBEDDING_DIMENSIONS."
    )

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        # No API keys required since this runs entirely on your local machine!
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': 'cpu'}  # Swaps to 'cuda' if you have a GPU
        )
    return _embeddings



def ensure_index():
    """Connect to the configured Pinecone index safely.

    Never automatically delete an existing index.
    Index migrations must be explicit administrative actions.
    """
    if not settings.pinecone_api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is missing."
        )

    desired_dimension = get_embedding_dimension()

    pc = Pinecone(
        api_key=settings.pinecone_api_key
    )

    index_names = {
        item["name"]
        for item in pc.list_indexes()
    }

    if settings.pinecone_index_name in index_names:
        index_info = pc.describe_index(
            settings.pinecone_index_name
        )

        current_dimension = getattr(
            index_info, "dimension", None
        )

        if current_dimension is None and isinstance(
            index_info, dict
        ):
            current_dimension = index_info.get(
                "dimension"
            )

        if current_dimension != desired_dimension:
            raise RuntimeError(
                "Pinecone embedding dimension mismatch. "
                f"Index '{settings.pinecone_index_name}' "
                f"has dimension {current_dimension}, but "
                f"the configured embedding model expects "
                f"{desired_dimension}. No index was deleted. "
                "Verify the model and index before migrating."
            )

        return pc.Index(
            settings.pinecone_index_name
        )

    # Create an index only if it does not already exist.
    pc.create_index(
        name=settings.pinecone_index_name,
        dimension=desired_dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )

    # Wait for the newly created index to become ready.
    import time

    deadline = time.monotonic() + 120

    while time.monotonic() < deadline:
        index_info = pc.describe_index(
            settings.pinecone_index_name
        )

        status = getattr(
            index_info, "status", None
        )

        if isinstance(status, dict):
            ready = status.get("ready", False)
        else:
            ready = getattr(
                status, "ready", False
            )

        if ready:
            return pc.Index(
                settings.pinecone_index_name
            )

        time.sleep(2)

    raise TimeoutError(
        "Pinecone index was created but did not "
        "become ready within 120 seconds."
    )



def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        index = ensure_index()
        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )
    return _vectorstore


def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": settings.top_k})



def add_documents(chunks):
    """
    Store document chunks using deterministic IDs.

    Re-ingesting an unchanged document with the same chunking
    configuration will update the corresponding vector IDs
    instead of creating new IDs for the same chunks.
    """
    if not chunks:
        return []

    store = get_vectorstore()
    ids = []

    for index, chunk in enumerate(chunks):
        document_hash = chunk.metadata.get("document_hash")

        if not document_hash:
            raise ValueError(
                "Missing document_hash in chunk metadata. "
                "Calculate the document hash before indexing."
            )

        document_name = (
            chunk.metadata.get("document_name")
            or Path(chunk.metadata.get("source", "unknown")).name
            or "unknown"
        )

        chunk_id = calculate_chunk_id(
            document_hash=document_hash,
            chunk_index=index,
            chunk_text=chunk.page_content,
        )

        metadata = build_chunk_metadata(
            document_name=document_name,
            document_hash=document_hash,
            chunk_index=index,
            chunk_text=chunk.page_content,
            extra_metadata=chunk.metadata,
        )

        # Preserve LangChain's document structure while adding
        # identity metadata needed for traceability.
        chunk.metadata = metadata   
        ids.append(chunk_id)

    return store.add_documents(documents=chunks, ids=ids)



def delete_documents(ids: list[str]):
    """
    Delete specific vectors from the configured Pinecone namespace.

    Only explicitly supplied IDs are deleted.
    """
    if not ids:
        return None

    index = ensure_index()

    return index.delete(
        ids=ids,
        namespace=settings.pinecone_namespace,
    )
