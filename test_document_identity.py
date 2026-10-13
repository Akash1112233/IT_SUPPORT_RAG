
from app.services.document_identity import (
    calculate_document_hash,
    calculate_chunk_id,
    build_chunk_metadata,
)


document = b"How to reset a company VPN connection."

# Test 1: Identical contents should have identical hashes.
hash_one = calculate_document_hash(document)
hash_two = calculate_document_hash(document)

assert hash_one == hash_two
print("PASS: Document hashes are stable.")

# Test 2: Changed contents should produce a different hash.
changed_hash = calculate_document_hash(
    b"How to configure a company VPN connection."
)

assert hash_one != changed_hash
print("PASS: Changed document contents produce a new hash.")

# Test 3: The same chunk identity should produce the same ID.
chunk_text = "Open Settings and select Network."

chunk_id_one = calculate_chunk_id(hash_one, 0, chunk_text)
chunk_id_two = calculate_chunk_id(hash_one, 0, chunk_text)

assert chunk_id_one == chunk_id_two
print("PASS: Chunk IDs are deterministic.")

# Test 4: Different chunks should have different IDs.
different_chunk_id = calculate_chunk_id(
    hash_one, 1, chunk_text
)

assert chunk_id_one != different_chunk_id
print("PASS: Different chunk positions have different IDs.")

# Test 5: Metadata should contain document identity.
metadata = build_chunk_metadata(
    document_name="vpn_troubleshooting.pdf",
    document_hash=hash_one,
    chunk_index=0,
    chunk_text=chunk_text,
    extra_metadata={"source_type": "internal_kb"},
)

assert metadata["document_name"] == "vpn_troubleshooting.pdf"
assert metadata["chunk_index"] == 0
assert metadata["source_type"] == "internal_kb"

print("PASS: Chunk metadata is created correctly.")
print("All document identity tests passed.")
