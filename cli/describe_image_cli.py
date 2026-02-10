import argparse
from describe_image import *

def main():
    parser = argparse.ArgumentParser(description="Describe image CLI")
    parser.add_argument("--image", type=str, help="Path to image")
    parser.add_argument("--query", type=str, help="Query for image")
    args = parser.parse_args()
    
    description = describe_image(args.image, args.query)
    print(f"Rewritten query: {description.text.strip()}")
    if description.usage_metadata is not None:
        print(f"Total tokens: {description.usage_metadata.total_token_count}")
    else:
        print("No usage metadata available")

if __name__ == "__main__":
    main()