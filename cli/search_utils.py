import json
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

DEFAULT_SEARCH_LIMIT = 5
DEFAULT_ALPHA = 0.5
BM25_K1 = 1.5
BM25_B = 0.75
SCORE_PRECISION = 3
DOCUMENT_PREVIEW_LENGTH = 100
RRF_K = 60
SEARCH_MULTIPLIER = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "rag-search-engine", "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "rag-search-engine", "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "rag-search-engine", "cache")

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError ("No API key set")

client = genai.Client(api_key=api_key)

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

def gemini_call(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

def gemini_call_no_safety(prompt: str) -> str:
    response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                    ]
                )          
            )
    return response.text

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