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
    rrf_search_parser.add_argument(
        "--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Rerank method",
    )
    rrf_search_parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate search results for relevance"
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
            results = rrf_search_command(args.query, args.k, args.limit, args.enhance, args.rerank_method, args.evaluate)
            movie_list = results['results']
            print(
                f"RRF Hybrid Search Results: '{results['original_query']}' (k={results['k']}):"
            )
            if args.evaluate and results.get('eval_scores'):
                print("Final evaluation report:")
                scores = results['eval_scores']
                for i, res in enumerate(movie_list):
                    score = scores[i] if i < len(scores) else 0
                    print(f"{i+1}. {res['title']}: {score}/3")
                    print("")            
            else:
                for i, res in enumerate(movie_list, 1):
                    print(
                        f"{i}. {res['title']}" 
                    )
                    if args.rerank_method == "individual" and 'rerank_score' in res:
                        print(f"    Rerank Score: {res['rerank_score']:.3f}/10")
                    elif args.rerank_method == "batch":
                        print(f"   Rerank Rank: {res.get('rerank_rank', 'N/A')}")
                    elif args.rerank_method == "cross_encoder":
                        print(f"   Cross Encoder Score: {res.get('cross_encoder_score', 0.0):.3f}")
                    print(f"RRF Score: {res['rrf_score']:.3f}")
                    bm_rank = res.get('bm25_rank', 'N/A')
                    sem_rank = res.get('semantic_rank', 'N/A')
                    if bm_rank != 'N/A' and sem_rank != 'N/A':
                        print(
                            f" BM25 Rank: {bm_rank}, Semantic Rank: {sem_rank}"
                        )
                    print(f"     {res['document'][:100]}")
                    print("")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()