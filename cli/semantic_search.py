import sys
import re
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
from search_utils import SCORE_PRECISION, load_movies

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("cannon generate embedding for empty text")
        return self.model.encode([text])[0]
        

    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        movie_strings = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=False)
        os.makedirs('cache', exist_ok=True)
        np.save('cache/embeddings.npy', self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}
        cache_path = 'cache/movie_embeddings.npy'
        if os.path.exists(cache_path):
            self.embeddings = np.load(cache_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search (self, query, limit):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError("No embeddings loaded. Call 'load or create embeddings' first.")
        if self.documents is None or self.documents.size == 0:
            raise ValueError("No documents loaded.  Call 'load_or_create_embeddings' first.")
        query_vec = self.generate_embedding(query)
        results = []
        for i, doc_embedding in enumerate(self.embeddings):
            score = cosine_similarity(query_vec, doc_embedding)
            results.append((score, self.documents[i]))
        results.sort(key=lambda x: x[0], reverse=True)
        top_results = []
        for score, doc in results[:limit]:
            top_results.append(
                {
                    'score': score,
                    'title': doc['title'],
                    'description': doc['description'],
                }
            )
        return top_results

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}
        all_chunk_strings = []
        all_chunk_metadata = []
        for movie_idx, doc in enumerate(documents):
            description = doc.get('description', "").strip()
            if not description:
                continue
            chunks = semantic_chunk(description, 4, 1)
            for chunk_idx, chunk_text in enumerate(chunks):
                all_chunk_strings.append(chunk_text)
                all_chunk_metadata.append({
                    'movie_idx': doc['id'],
                    'chunk_idx': chunk_idx,
                    'total_chunks': len(chunks),
                })
        self.chunk_embeddings = self.model.encode(all_chunk_strings, show_progress_bar=False)
        self.chunk_metadata = all_chunk_metadata
        os.makedirs('cache', exist_ok=True)
        np.save('cache/chunk_embeddings.npy', self.chunk_embeddings)
        np.save('cache/chunk_metadata.npy', all_chunk_metadata)
        with open('cache/chunk_metadata.json', 'w') as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(all_chunk_strings)}, f, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}
        embed_path = 'cache/chunk_embeddings.npy'
        metadata_path = 'cache/chunk_metadata.json'
        if os.path.exists(embed_path) and os.path.exists(metadata_path):
            self.chunk_embeddings = np.load(embed_path)
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None:
            raise ValueError("No chunk embeddings loaded.  Call 'embed_chunks' first.")
        query_vec = self.generate_embedding(query)
        best_movie_scores = {}
        for i, chunk_vec in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_vec, chunk_vec)
            m_idx = self.chunk_metadata[i]['movie_idx']
            if m_idx not in best_movie_scores or score > best_movie_scores[m_idx][0]:
                best_movie_scores[m_idx] = (score, i)
        sorted_movies = sorted(
            best_movie_scores.items(),
            key=lambda x: x[1][0],
            reverse=True,
        )
        top_results = []
        for m_idx, (score, chunk_idx) in sorted_movies[:limit]:
            top_results.append({
                "id": m_idx,
                "title": self.document_map[m_idx]["title"],
                "document": self.document_map[m_idx]["description"][:100],
                "score": round(score, SCORE_PRECISION),
                "metadata": self.chunk_metadata[chunk_idx],
            })
        return top_results
        

def search_chunked(query, limit):
    search_instance = ChunkedSemanticSearch()
    documents = load_movies()
    embeddings = search_instance.load_or_create_chunk_embeddings(documents)
    results = search_instance.search_chunks(query, limit)
    for i, res in enumerate(results, 1):
        print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
        print(f"   {res['document']}...")

def embed_chunks():
    search_instance = ChunkedSemanticSearch()
    documents = load_movies()
    embeddings = search_instance.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")
    return embeddings

def chunk(text, chunk_size, overlap):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")

def semantic_chunk(text, max_chunk_size, overlap):
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) == 1 and not re.search(f'[.!?]$', sentences[0]):
        cleaned_sentences = [sentences[0]]
    else:
        cleaned_sentences = [s.strip() for s in sentences if s.strip()]
    if not cleaned_sentences:
        return []
    chunks = []
    step = max_chunk_size - overlap
    for i in range(0, len(sentences), step):
        chunk_slice = sentences[i : i + max_chunk_size]
        if chunk_slice:
            chunks.append(" ".join(chunk_slice))
        if i + max_chunk_size >= len(sentences):
            break
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")
    return chunks

def search(query, limit):
    search_instance = SemanticSearch()
    documents = load_movies()
    embeddings = search_instance.load_or_create_embeddings(documents)
    results = search_instance.search(query, limit)
    for i, res in enumerate(results, 1):
        title = res['title']
        score = res['score']
        description = res['description']
        print(f"{i}. {title} (score: {score:.4f})")
        print(f"   {description[:100]}...")
        print()

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def embed_query_text(query):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def verify_embeddings():
    search_instance = SemanticSearch()
    documents = load_movies()
    embeddings = search_instance.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )

def embed_text(text):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_model():
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print (f"Max sequence length: {search_instance.model.max_seq_length}")
    