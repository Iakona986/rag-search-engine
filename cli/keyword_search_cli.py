#!/usr/bin/env python3
import sys
import argparse

from keyword_search import *
from search_utils import BM25_K1, BM25_B


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")
    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document and term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")
    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term")
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF for a document and term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term")
    bm25idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF for a term")
    bm25idf_parser.add_argument("term", type=str, help="Term")
    bm25tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF for a document and term")
    bm25tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25tf_parser.add_argument("term", type=str, help="Term")
    bm25tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 B parameter")

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")
        case "tf":
            doc_id = args.doc_id
            term = args.term
            tf = tf_command(doc_id, term)
            print(f"TF score of '{term}' in document '{doc_id}': {tf:.2f}")
        case "idf":
            term = args.term
            idf = idf_command(term)
            print(f"IDF score of '{term}': {idf:.2f}")
        case "tfidf":
            doc_id = args.doc_id
            term = args.term
            tfidf = tfidf_command(doc_id, term)
            print(f"TF-IDF score of '{term}' in document '{doc_id}': {tfidf:.2f}")
        case "bm25idf":
            term = args.term
            bm25idf = bm25idf_command(term)
            print(f"BM25 IDF score of '{term}': {bm25idf:.2f}")
        case "bm25tf":
            doc_id = args.doc_id
            term = args.term
            k1 = args.k1
            bm25tf = bm25_tf_command(doc_id, term, k1)
            print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25tf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
