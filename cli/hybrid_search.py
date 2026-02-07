import os

from keyword_search import InvertedIndex
from semantic_search import ChunkedSemanticSearch
from sentence_transformers import CrossEncoder
from search_utils import (
    DEFAULT_ALPHA,
    DEFAULT_SEARCH_LIMIT,
    format_search_results,
    load_movies
)
from dotenv import load_dotenv
from google import genai
from google.genai import types
import time
import logging
import json

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError ("No API key set")

client = genai.Client(api_key=api_key)

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

    def rrf_search(self, query, k, limit=10, rerank_method=None):
        search_limit = limit * 5 if rerank_method else limit
        act_limit = search_limit * 500
        bm25_results = self._bm25_search(query, act_limit)
        semantic_results = self.semantic_search.search_chunks(query, act_limit)
        pairs = []
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
        results = sorted(rrf_combined.values(), key=lambda x: x['rrf_score'], reverse=True)[:search_limit]
        logger.debug(f"Results after RRF Search: {rrf_combined.values()}")
        if rerank_method == "individual":
            print(f"Reranking top {search_limit} results using individual method...")
            for doc in results:
                doc['rerank_score'] = self.get_individual_score(query, doc)
                time.sleep(3)
            results = sorted(results, key=lambda x: x.get('rerank_score', 0), reverse=True)
            logger.debug(f"Results after Individual Reranking: {results}")
        elif rerank_method == "batch":
            print(f"Reranking top {search_limit} results using batch method...")
            doc_list_str = ""
            for i, doc in enumerate(results, 1):
                doc_list_str += f"{i}. {doc['title']} - {doc['document'][:100]}"
            prompt = f"""Rank these movies by relevance to the search query.

            Query: "{query}"

            Movies:
            {doc_list_str}

            Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

            [75, 12, 34, 2, 1]
            """
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            try:
                clean_json = response.text.strip().replace("```json", "").replace("```", "")
                reranked_ids = json.loads(clean_json)
                rank_mapping = {int(doc_id): rank for rank, doc_id in enumerate(reranked_ids, 1)}
                for i, doc in enumerate(results):
                    doc['rerank_rank'] = rank.mapping.get(i, 999)
                results.sort(key=lambda x: x.get('rerank_rank', 999))
                logger.debug(f"Results after Batch Reranking: {results}")
            except Exception as e:
                print(f"Error parsing batch rerank JSON: {e}")
        elif rerank_method == "cross_encoder":
            print(f"Reranking top {search_limit} results using cross encoder method...")
            pairs = []
            for doc in results:
                pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
            scores = cross_encoder.predict(pairs)
            for i, score in enumerate(scores):
                results[i]['cross_encoder_score'] = float(score)
            results.sort(key=lambda x: x.get('cross_encoder_score', 0), reverse=True)
            logger.debug(f"Results after Cross-Encoder Reranking: {results}")
        return results[:limit]
    
    def get_individual_score(self, query, doc):
        prompt = f"""Rate how well this movie matches the search query.

            Query: "{query}"
            Movie: {doc.get("title", "")} - {doc.get("document", "")}

            Consider:
            - Direct relevance to query
            - User intent (what they're looking for)
            - Content appropriateness

            Rate 0-10 (10 = perfect match).
            Give me ONLY the number in your response, no other text or explanation.

            Score:"""
        try:
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
            if response.text:
                score_str = response.text.strip()
                return float(score_str)
            print(f"Warning: No text returned for {doc.get('title')}. Likely blocked by safety filters.")
            return 0.0
        except Exception as e:
            print(f"Error getting individual score: {e}")
            return 0.0

    def evaluate_results(self, query, results):
        formatted_results = [
            f"{i+1}. {result['title']} - {result['document'][:200]}"
            for i, result in enumerate(results)
        ]
        prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

                Query: "{query}"

                Results:
                {chr(10).join(formatted_results)}

                Scale:
                - 3: Highly relevant
                - 2: Relevant
                - 1: Marginally relevant
                - 0: Not relevant

                Do NOT give any numbers out than 0, 1, 2, or 3.

                Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

                [2, 0, 3, 2, 0, 1]"""
            
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            clean_json = response.text.strip().replace("```json", "").replace("```", "")
            scores = json.loads(clean_json)
            return scores
        except Exception as e:
            print(f"Error evaluating results: {e}")
            return None
            

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
    query: str, k: int = 60, limit: int = DEFAULT_SEARCH_LIMIT, enhance: str | None = "", rerank: str | None = "", evaluate: bool = False
) -> list[dict]:
    movies = load_movies()
    searcher = HybridSearch(movies)
    logger.debug(f"Original query: '{query}'")
    match enhance:
        case "spell":
            messages = f"""Fix any spelling errors in this movie search query.

                Only correct obvious typos. Don't change correctly spelled words.

                Query: "{query}"

                If no errors, return the original query. Do not capitalize any uncapitalized words.
                """
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=messages,
            )
            time.sleep(3)
            logger.debug(f"Enhanced query ({enhance}): '{query}' -> '{response.text}'")
            query = response.text
        case "rewrite":
            messages = f"""Rewrite this movie search query to be more specific and searchable.

                Original: "{query}"

                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions (horror = scary, animation = cartoon)
                - Keep it concise (under 10 words)
                - It should be a google style search query that's very specific
                - Don't use boolean logic

                Examples:

                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                """
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=messages,
            )
            time.sleep(3)
            print(f"Enhanced query ({enhance}): '{query}' -> '{response.text}'\n")
            query = response.text
        case "expand":
            messages = f"""Expand this movie search query to be more specific and searchable.

                Original: "{query}"

                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions (horror = scary, animation = cartoon)
                - Keep it concise (under 10 words)
                - It should be a google style search query that's very specific
                - Don't use boolean logic

                Examples:

                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                """
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=messages,
            )
            time.sleep(3)
            print(f"Enhanced query ({enhance}): '{query}' -> '{response.text}'\n")
            query = response.text
        case _:
            pass
    original_query = query
    search_limit = limit
    results = searcher.rrf_search(query, k, search_limit, rerank_method=rerank)
    eval_scores = []
    if evaluate:
        eval_scores = searcher.evaluate_results(query, results)
    return {
        "original_query": original_query,
        "k": k,
        "limit": limit,
        "results": results,
        "eval_scores": eval_scores,
    }
    
def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)