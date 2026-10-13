
from app.services.document_manifest import (
    init_manifest,
    get_document,
    is_document_unchanged,
    mark_document_indexed,
    mark_document_pending,
)


def run_tests():
    init_manifest()
    print("PASS: Manifest database initialized.")

    # Use a unique test name so an existing document is not overwritten.
    name = "__manifest_test_v1__.txt"
    hash_v1 = "a" * 64
    hash_v2 = "b" * 64

    # A new document should not appear indexed.
    assert not is_document_unchanged(name, hash_v1)
    print("PASS: Unknown documents are not treated as indexed.")

    # Simulate a successfully indexed document.
    ids = ["test-chunk-001", "test-chunk-002"]
    mark_document_indexed(name, hash_v1, ids)

    record = get_document(name)

    assert record is not None
    assert record["status"] == "indexed"
    assert record["chunk_count"] == 2
    assert record["chunk_ids"] == ids
    print("PASS: Indexed document metadata is saved.")

    # Matching content should be recognized.
    assert is_document_unchanged(name, hash_v1)
    print("PASS: Unchanged document versions are detected.")

    # Changed content should not be treated as unchanged.
    assert not is_document_unchanged(name, hash_v2)
    print("PASS: Changed document versions are detected.")

    # Pending status must not be mistaken for a completed index.
    mark_document_pending(name, hash_v2)

    pending = get_document(name)
    assert pending is not None
    assert pending["status"] == "pending"
    assert not is_document_unchanged(name, hash_v2)
    print("PASS: Pending versions are not marked as indexed.")

    # Restore the original test record for a repeatable test.
    # This only changes the local test manifest, not Pinecone.
    mark_document_indexed(name, hash_v1, ids)

    print("\nAll document manifest tests passed.")
    print("No Pinecone writes were performed.")


if __name__ == "__main__":
    run_tests()
