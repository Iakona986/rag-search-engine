import os

from keyword_search import InvertedIndex
from semantic_search import ChunkedSemanticSearch
from search_utils import (
    DEFAULT_ALPHA,
    DEFAULT_SEARCH_LIMIT,
    format_search_results,
    load_movies
)


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha: float, limit: int = 5) -> list[dict]:
        act_limit = limit * 500
        bm25_results = self._bm25_search(query, act_limit)
        semantic_results = self.semantic_search.search_chunks(query, act_limit)
        combined = combine_search_results(bm25_results, semantic_results, alpha)
        return combined[:limit]      

    def rrf_search(self, query, k, limit=10):
        act_limit = limit * 500
        bm25_results = self._bm25_search(query, act_limit)
        semantic_results = self.semantic_search.search_chunks(query, act_limit)
        rrf_combined = {}
        for rank, res in enumerate(bm25_results, 1):
            doc_id = res['id']
            rrf_combined[doc_id] = {
                'title': res['title'],
                'document': res['document'],
                'bm25_rank': rank,
                'semantic_rank': None,
                'rrf_score': rrf_score(rank, k),
            }
        for rank, res in enumerate(semantic_results, 1):
            doc_id = res['id']
            score = rrf_score(rank, k)
            if doc_id in rrf_combined:
                rrf_combined[doc_id]['semantic_rank'] = rank
                rrf_combined[doc_id]['rrf_score'] += score
            else:
                rrf_combined[doc_id] = {
                    'title': res['title'],
                    'document': res['document'],
                    'bm25_rank': None,
                    'semantic_rank': rank,
                    'rrf_score': score,
                }
        return sorted(rrf_combined.values(), key=lambda x: x['rrf_score'], reverse=True)[:limit]

def normalize(scores: list[float]):
    normalized_scores = []
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    for s in scores:
        normalized_scores.append((s - min_score) / (max_score - min_score))
    return normalized_scores

def normalize_search_results(results: list[dict]) -> list[dict]:
    scores: list[float] = []
    for result in results:
        scores.append(result["score"])
    normalized: list[float] = normalize(scores)
    for i, result in enumerate(results):
        result["normalized_score"] = normalized[i]
    return results

def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = DEFAULT_ALPHA
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def combine_search_results(
    bm25_results: list[dict], semantic_results: list[dict], alpha: float = DEFAULT_ALPHA
) -> list[dict]:
    bm25_norm = normalize_search_results(bm25_results)
    semantic_norm = normalize_search_results(semantic_results)
    combined_scores = {}
    for result in bm25_norm:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["bm25_score"]:
            combined_scores[doc_id]["bm25_score"] = result["normalized_score"]
    
    for result in semantic_norm:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["semantic_score"]:
            combined_scores[doc_id]["semantic_score"] = result["normalized_score"]
    hybrid_results = []
    for doc_id, data in combined_scores.items():
        score_value = hybrid_score(data["bm25_score"], data["semantic_score"], alpha)
        result = format_search_results(
            doc_id = doc_id,
            title = data["title"],
            document = data["document"],
            score = score_value,
            bm25_score = data["bm25_score"],
            semantic_score = data["semantic_score"],
        )
        hybrid_results.append(result)
    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)

def weighted_search_command(
    query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[dict]:
    movies = load_movies()
    searcher = HybridSearch(movies)
    original_query = query
    search_limit = limit
    results = searcher.weighted_search(query, alpha, search_limit)
    return {
        "original_query": original_query,
        "alpha": alpha,
        "limit": limit,
        "results": results,
    }

def rrf_search_command(
    query: str, k: int = 60, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[dict]:
    movies = load_movies()
    searcher = HybridSearch(movies)
    original_query = query
    search_limit = limit
    results = searcher.rrf_search(query, k, search_limit)
    return {
        "original_query": original_query,
        "k": k,
        "limit": limit,
        "results": results,
    }
    
def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)