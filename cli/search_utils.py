import json
import os

DEFAULT_SEARCH_LIMIT = 5
DEFAULT_ALPHA = 0.5
BM25_K1 = 1.5
BM25_B = 0.75
SCORE_PRECISION = 3
DOCUMENT_PREVIEW_LENGTH = 100

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "rag-search-engine", "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "rag-search-engine", "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "rag-search-engine", "cache")


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_golden() -> list[dict]:
    with open("data/golden_dataset.json", "r") as f:
        data = json.load(f)
    return data

def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()

def format_search_results(
    doc_id: str, title: str, document: str, score: float, **metadata: Any
) -> dict[str, Any]:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }