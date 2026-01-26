import json
import os

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "rag-search-engine", "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "rag-search-engine", "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "rag-search-engine", "cache")


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()
