
from app.rag.vectorstore import search_documents_with_scores


# Edit expected_terms to match the actual content of your KB.
# Each query passes if at least one of its expected terms is found
# in the returned document names or chunk text.
BENCHMARK = [
    {
        "query": "How do I troubleshoot a VPN connection?",
        "expected_terms": ["vpn"],
    },
    {
        "query": "How do I troubleshoot email problems?",
        "expected_terms": ["email"],
    },
    {
        "query": "How do I install company software?",
        "expected_terms": ["software", "install"],
    },
    {
        "query": "How do I fix a laptop Wi-Fi connection?",
        # Change this to ["wifi"] if you add a Wi-Fi guide later.
        "expected_terms": ["wifi", "wi-fi"],
        "allow_no_match": True,
    },
]


def main():
    hits = 0
    total = len(BENCHMARK)

    for case in BENCHMARK:
        query = case["query"]
        expected_terms = [
            term.lower() for term in case["expected_terms"]
        ]

        print("\n" + "=" * 72)
        print("QUERY:", query)

        results = search_documents_with_scores(query, k=4)

        found = False

        for rank, result in enumerate(results, start=1):
            metadata = result["metadata"]
            document_name = str(
                metadata.get(
                    "document_name",
                    metadata.get("source", "unknown"),
                )
            )

            searchable_text = (
                document_name + " " + result["content"]
            ).lower()

            is_match = any(
                term in searchable_text
                for term in expected_terms
            )

            if is_match:
                found = True

            print(
                f"\nRank: {rank} | Score: {result['score']:.4f}"
            )
            print("Document:", document_name)
            print("Expected-term match:", is_match)
            print("Preview:", result["content"][:180])

        if found:
            hits += 1
            print("\nRESULT: HIT")
        elif case.get("allow_no_match", False):
            print("\nRESULT: NO MATCH (allowed for this query)")
        else:
            print("\nRESULT: MISS")

    # The Wi-Fi case is an intentional negative/coverage check,
    # so calculate the score only over the first three cases.
    scored_cases = [
        case for case in BENCHMARK
        if not case.get("allow_no_match", False)
    ]

    scored_hits = 0

    for case in scored_cases:
        results = search_documents_with_scores(case["query"], k=4)
        terms = [term.lower() for term in case["expected_terms"]]

        if any(
            any(
                term in (
                    str(result["metadata"].get("document_name", ""))
                    + " "
                    + result["content"]
                ).lower()
                for term in terms
            )
            for result in results
        ):
            scored_hits += 1

    hit_rate = (
        scored_hits / len(scored_cases)
        if scored_cases else 0.0
    )

    print("\n" + "=" * 72)
    print(f"Hit Rate@4: {hit_rate:.1%}")
    print(
        "Note: this is a simple term-based diagnostic, "
        "not a semantic relevance evaluation."
    )


if __name__ == "__main__":
    main()
