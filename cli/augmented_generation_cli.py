import argparse
from search_utils import load_movies, gemini_call
from hybrid_search import HybridSearch

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            movies = load_movies()
            searcher = HybridSearch(movies)
            movies_list = searcher.rrf_search(query)
            context_docs = "\n".join([f"- {m['title']}: {m['document']}" for m in movies_list])
            prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                Query: {query}

                Documents:
                {context_docs}

                Provide a comprehensive answer that addresses the query:"""
            answer = gemini_call(prompt)
            print("Search Results:")
            for res in movies_list:
                print(f"    - {res['title']}")
            print("\n")
            print("RAG Response:")
            print(answer)
            print("\n")            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()