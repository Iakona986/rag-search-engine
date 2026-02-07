import argparse
from search_utils import load_golden, load_movies
from hybrid_search import HybridSearch


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    top_k = args.limit
    k = 60
    movies = load_movies()
    golden = load_golden()
    search = HybridSearch(movies)
    for test_case in golden["test_cases"]:
        query = test_case["query"]
        relevant_titles = test_case["relevant_docs"]
        result = search.rrf_search(query, k, top_k)
        relevant_docs = [res["title"] for res in result]
        matched_docs = [title for title in relevant_docs if title in relevant_titles]
        precision = len(matched_docs) / len(relevant_docs)
        recall = len(matched_docs) / len(relevant_titles) 
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"- Query: {query}")
        print(f"    - Precision@{top_k}: {precision:.4f}")
        print(f"    - Recall@{top_k}: {recall:.4f}")
        print(f"    - F1 Score: {f1:.4f}")
        print(f"    - Retrieved: {relevant_docs}")
        print(f"    - Relevant: {relevant_titles}")
        print("\n")
    


if __name__ == "__main__":
    main()