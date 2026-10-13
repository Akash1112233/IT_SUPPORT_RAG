
from app.rag.vectorstore import inspect_retrieval_results


def main():
    queries = [
        "How do I troubleshoot a VPN connection?",
        "How can I reset my company password?",
        "What should I do if my laptop cannot connect to Wi-Fi?",
    ]

    for query in queries:
        print("\n" + "=" * 70)
        print("QUERY:", query)
        print("=" * 70)

        try:
            results = inspect_retrieval_results(query, k=4)
        except Exception:
            print("Retrieval failed. Check the traceback in the terminal.")
            raise

        if not results:
            print("No results returned.")
            continue

        for item in results:
            print(f"\nRank: {item['rank']}")
            print(f"Score: {item['score']:.4f}")
            print(f"Document: {item['document_name']}")
            print(f"Chunk index: {item['chunk_index']}")
            print(f"Preview: {item['content_preview']}")

    print("\nDiagnostic run complete.")


if __name__ == "__main__":
    main()
