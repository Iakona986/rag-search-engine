import sys
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if isinstance(text, list):
            if not text:
                raise ValueError("List must not be empty")
            if any(not isinstance(t, str) or t.strip() == "" for t in text):
                raise ValueError("List must contain only non-empty strings")
            return self.model.encode(text, show_progress_bar=False)
        else:
            if not isinstance(text, str) or text.strip() == "":
                raise ValueError("Text must not be empty")
            return self.model.encode(text)
        

    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}
        movie_strings = [f"{doc['title']}: {doc['description']}" for doc in documents]
        print("Generating embeddings...")
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
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_vec = self.generate_embedding(query)
        results = []
        for i, doc_vec in enumerate(self.embeddings):
            score = cosine_similarity(query_vec, doc_vec)
            results.append((score, self.documents[i]))
        results.sort(key=lambda x: x[0], reverse=True)
        top_results = []
        for score, doc in results[:limit]:
            top_results.append({
                "score": score,
                "title": doc["title"],
                "description": doc["description"],
            })
        return top_results

def chunk(text, chunk_size, overlap):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")

def search(query, limit):
    search_instance = SemanticSearch()
    with open('data/movies.json', 'r') as f:
        data = json.load(f)
    documents = data["movies"]
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
    with open('data/movies.json', 'r') as f:
        data = json.load(f)
    documents = data["movies"]
    embeddings = search_instance.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

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