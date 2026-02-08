import argparse
from augmented_generation import *

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument(
        "query", type=str, help="Search query for RAG"
    )

    summarize_parser = subparsers.add_parser(
        "summarize", help="Use LLM to generate a summary of the documents pulled from the search."
    )
    summarize_parser.add_argument(
        "query", type=str, help="Search query for summary"
    )
    summarize_parser.add_argument(
         "--limit", type=int, default=5, help="Number of documents to retrieve"
    )

    citations_parser = subparsers.add_parser(
        "citations", help="Use LLM to generate a cited information on the answers to the query."
    )   
    citations_parser.add_argument(
        "query", type=str, help="Search query for citation"
    )
    citations_parser.add_argument(
        "--limit", type=int, default=5, help="Number of documents to retrieve"
    )

    question_parser = subparsers.add_parser(
        "question", help="Use LLM to answer a question based on the documents pulled from the search."
    )
    question_parser.add_argument(
        "query", type=str, help="Search query for question"
    )
    question_parser.add_argument(
        "--limit", type=int, default=5, help="Number of documents to retrieve"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            movies_list, answer = rag_command(args.query)
            print("Search Results:")
            for res in movies_list:
                print(f"    - {res['title']}")
            print()
            print("RAG Response:")
            print(answer)
            print()
        case "summarize":
            results, summary = summarize_command(args.query, args.limit)
            print("Search Results:")
            for res in results:
                print(f"    - {res['title']}")
            print()
            print("LLM Summary:")
            print(summary)
            print()
        case "citations":
            results, answer = citations_command(args.query, args.limit)
            print("Search Results:")
            for res in results:
                print(f"    - {res['title']}")
            print()
            print("LLM Answer:")
            print(answer)
            print()
        case "question":
            results, answer = question_command(args.query, args.limit)
            print("Search Results:")
            for res in results:
                print(f"    - {res['title']}")
            print()
            print("LLM Answer:")
            print(answer)
            print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()