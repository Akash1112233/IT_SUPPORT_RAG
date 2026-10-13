
from app.rag.vectorstore import search_documents_with_scores


def main():
    query = (
        "What should an employee do when the company "
        "VPN connection fails?"
    )

    print(f"\nQuery: {query}\n")

    results = search_documents_with_scores(query, k=4)

    if not results:
        print("No matching documents were returned.")
        return

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Score: {result['score']:.4f}")
        print(f"Metadata: {result['metadata']}")
        print(f"Content:\n{result['content']}")

    matching = [
        result for result in results
        if "vpn troubleshooting" in result["content"].lower()
        or "vpn application" in result["content"].lower()
    ]

    print("\n--- Evaluation ---")
    print(f"Results returned: {len(results)}")
    print(f"Relevant-looking results: {len(matching)}")

    if matching:
        print("PASS: At least one relevant-looking result was retrieved.")
    else:
        print(
            "CHECK: No obvious VPN match found. "
            "Inspect the returned chunks and metadata."
        )


if __name__ == "__main__":
    main()
