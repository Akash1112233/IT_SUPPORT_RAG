
from pathlib import Path
from tempfile import TemporaryDirectory

from app.services.ingestion import load_file, chunk_documents
from app.services.document_identity import (
    calculate_document_hash,
    calculate_chunk_id,
    build_chunk_metadata,
)


def run_tests():
    with TemporaryDirectory() as temp_dir:
        # 1. Create a temporary sample IT support document.
        file_path = Path(temp_dir) / "vpn_troubleshooting.txt"

        content = (
            "VPN TROUBLESHOOTING GUIDE\n\n"
            "Step 1: Check your internet connection.\n\n"
            "Step 2: Open the company VPN application.\n\n"
            "Step 3: Sign in using your company credentials.\n\n"
            "Step 4: Restart the VPN application if the connection fails.\n\n"
            "Step 5: Contact IT support if the problem continues.\n"
        )

        original_bytes = content.encode("utf-8")
        file_path.write_bytes(original_bytes)

        # 2. Calculate the document fingerprint.
        document_hash = calculate_document_hash(original_bytes)

        assert len(document_hash) == 64
        assert document_hash == calculate_document_hash(original_bytes)
        print("PASS: Document SHA-256 hash generated.")

        # 3. Load the document using the existing loader.
        documents = load_file(file_path)

        assert len(documents) > 0
        assert "VPN TROUBLESHOOTING GUIDE" in documents[0].page_content
        print("PASS: Document loaded successfully.")

        # 4. Attach document identity before chunking.
        for document in documents:
            document.metadata.update({
                "document_name": file_path.name,
                "document_hash": document_hash,
            })

        # 5. Split the document using the existing chunker.
        chunks = chunk_documents(documents)

        assert len(chunks) > 0
        assert all(chunk.page_content.strip() for chunk in chunks)
        print(f"PASS: Document split into {len(chunks)} chunk(s).")

        # 6. Check identity and metadata for every chunk.
        chunk_ids = []

        for index, chunk in enumerate(chunks):
            assert chunk.metadata["document_hash"] == document_hash
            assert chunk.metadata["document_name"] == file_path.name

            chunk_id = calculate_chunk_id(
                document_hash=document_hash,
                chunk_index=index,
                chunk_text=chunk.page_content,
            )

            metadata = build_chunk_metadata(
                document_name=file_path.name,
                document_hash=document_hash,
                chunk_index=index,
                chunk_text=chunk.page_content,
                extra_metadata=chunk.metadata,
            )

            assert metadata["document_hash"] == document_hash
            assert metadata["chunk_index"] == index
            assert len(metadata["chunk_hash"]) == 64

            chunk_ids.append(chunk_id)

        print("PASS: Every chunk has identity metadata.")

        # 7. Confirm the generated IDs are unique in this document.
        assert len(chunk_ids) == len(set(chunk_ids))
        print("PASS: Chunk IDs are unique.")

        # 8. Confirm the same chunk gets the same ID on reprocessing.
        repeated_id = calculate_chunk_id(
            document_hash=document_hash,
            chunk_index=0,
            chunk_text=chunks[0].page_content,
        )

        assert repeated_id == chunk_ids[0]
        print("PASS: Chunk IDs remain stable on reprocessing.")

    print("\nAll local ingestion tests passed.")
    print("No Pinecone writes were performed.")


if __name__ == "__main__":
    run_tests()
