#!/usr/bin/env python3

import argparse
from semantic_search import *
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify that the embedding model is loaded")
    embed_parser = subparsers.add_parser("embed_text", help="Embed a text")
    embed_parser.add_argument("text", type=str, help="Text to embed")
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify that the embeddings are loaded")
    embedquery_parser = subparsers.add_parser("embedquery", help="Embed a query")
    embedquery_parser.add_argument("query", type=str, help="Query to embed")
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "embedquery":
            embed_query_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()