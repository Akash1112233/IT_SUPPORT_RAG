
from uuid import uuid4

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

    # A fresh name prevents records from previous test runs
    # from affecting the assertions.
    name = f"__manifest_test_{uuid4().hex}.txt"

    hash_v1 = "a" * 64
    hash_v2 = "b" * 64

    # Test 1: A new document should not appear indexed.
    assert not is_document_unchanged(name, hash_v1)
    print("PASS: Unknown documents are not treated as indexed.")

    # Test 2: Save a simulated indexed document.
    ids = ["test-chunk-001", "test-chunk-002"]
    mark_document_indexed(name, hash_v1, ids)

    record = get_document(name)

    assert record is not None
    assert record["status"] == "indexed"
    assert record["chunk_count"] == 2
    assert record["chunk_ids"] == ids
    print("PASS: Indexed document metadata is saved.")

    # Test 3: Matching content should be recognized.
    assert is_document_unchanged(name, hash_v1)
    print("PASS: Unchanged document versions are detected.")

    # Test 4: Changed content should not be treated as unchanged.
    assert not is_document_unchanged(name, hash_v2)
    print("PASS: Changed document versions are detected.")

    # Test 5: Pending status is not considered indexed.
    mark_document_pending(name, hash_v2)

    pending = get_document(name)

    assert pending is not None
    assert pending["status"] == "pending"
    assert not is_document_unchanged(name, hash_v2)
    print("PASS: Pending versions are not marked as indexed.")

    print("\nAll document manifest tests passed.")
    print("No Pinecone writes were performed.")


if __name__ == "__main__":
    run_tests()
