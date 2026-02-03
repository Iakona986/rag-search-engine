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
    search_parser = subparsers.add_parser("search", help="Search for a query")
    search_parser.add_argument("query", type=str, help="Query to search for")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    chunk_parser = subparsers.add_parser("chunk", help="Chunk a text")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Size of each chunk")
    chunk_parser.add_argument("--overlap", type=int, default=20, help="Number of words to overlap between chunks")
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk a text semantically")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Size of each chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of words to overlap between chunks")
    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Embed the chunks")
    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search for a query in the chunks")
    search_chunked_parser.add_argument("query", type=str, help="Query to search for")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
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
        case "search":
            search(args.query, args.limit)
        case "chunk":
            chunk(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks()
        case "search_chunked":
            search_chunked(args.query, args.limit)
        case _: 
            parser.print_help()

if __name__ == "__main__":
    main()