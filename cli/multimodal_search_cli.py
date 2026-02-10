import argparse
from multimodal_search import verify_image_embedding, image_search_command

def main():
    parser = argparse.ArgumentParser(description="Verify image embedding")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify_image_embedding", help="Verify image embedding")
    verify_parser.add_argument("image_path", type=str, help="Path to image")

    image_search_parser = subparsers.add_parser("image_search", help="Use an image to search movies DB")
    image_search_parser.add_argument("image_path", type=str, help="Path to image")
    args = parser.parse_args()
    match args.command:
        case "verify_image_embedding":
            embedding_shape = verify_image_embedding(args.image_path)
            print(f"Embedding shape: {embedding_shape[0]} dimensions")
        case "image_search":
            results = image_search_command(args.image_path)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']} (similarity: {res['score']})")
                print(f"    {res['description']:.100}...")
                print()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()