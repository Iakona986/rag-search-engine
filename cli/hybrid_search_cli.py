import argparse
from hybrid_search import *


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid Search CLI"
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )
    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize scores"
    )
    normalize_parser.add_argument(
        "scores", nargs="+", type=float, help="Scores to normalize"
    )
    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Weighted hybrid search"
    )
    weighted_search_parser.add_argument("query", type=str, help="Query to search for")
    weighted_search_parser.add_argument(
        "--alpha", type=float, default=0.5, help="Weight for semantic search"
    )
    weighted_search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )
    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="RRF hybrid search"
    )
    rrf_search_parser.add_argument("query", type=str, help="Query to search for")
    rrf_search_parser.add_argument(
        "-k", type=int, default=60, help="How heavily to weigh higher ranked results"
    )
    rrf_search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )
    rrf_search_parser.add_argument(
        "--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method",
    )
    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            results = weighted_search_command(args.query, args.alpha, args.limit)
            print(
                f"Weighted Hybrid Search Results: '{results['original_query']}' (alpha={results['alpha']}):"
            )
            print(
                f" Alpha {results['alpha']}, {int(results['alpha'] * 100)}% Keyword, {int((1 - results['alpha']) * 100)}% Semantic"
            )
            for i, res in enumerate(results['results'], 1):
                print(
                    f"{i}, {res['title']}\n Hybrid Score: {res['score']:.3f}" 
                )
                metadata = res.get('metadata', {})
                if "bm25_score" in metadata and "semantic_score" in metadata:
                    print(
                        f" BM25: {metadata['bm25_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}"
                    )
                print(f"     {res['document'][:100]}")
                print("")
        case "rrf-search":
            results = rrf_search_command(args.query, args.k, args.limit, args.enhance)
            print(
                f"RRF Hybrid Search Results: '{results['original_query']}' (k={results['k']}):"
            )
            for i, res in enumerate(results['results'], 1):
                print(
                    f"{i}. {res['title']}\n RRF Score: {res['rrf_score']:.3f}" 
                )
                metadata = res.get('metadata', {})
                if "bm25_rank" in metadata and "semantic_rank" in metadata:
                    print(
                        f" BM25 Rank: {metadata['bm25_rank']}, Semantic Rank: {metadata['semantic_rank']}"
                    )
                print(f"     {res['document'][:100]}")
                print("")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()