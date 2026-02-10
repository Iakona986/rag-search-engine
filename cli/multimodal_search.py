from PIL import Image
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from search_utils import load_movies, RRF_K, SCORE_PRECISION
from semantic_search import cosine_similarity
import math

class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.movies = load_movies()
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.movies]
        self.text_embeddings = self.model.encode(
            self.texts, 
            convert_to_tensor=True,
            show_progress_bar=True,
        )
    
    def embed_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path)
        embedding = self.model.encode([image], convert_to_tensor=True)
        return embedding

    def search_with_image(self, image_path):
        image_embedding = self.embed_image(image_path)
        if torch.is_tensor(image_embedding):
            image_embedding = image_embedding.detach().cpu().numpy()
        image_embedding = image_embedding.flatten().astype(np.float32)
        if torch.is_tensor(self.text_embeddings):
            texts_np = self.text_embeddings.detach().cpu().numpy()
        else:
            texts_np = self.text_embeddings
        similarities = []
        for i, text_embedding in enumerate(texts_np):
            similarity = cosine_similarity(text_embedding, image_embedding)
            similarities.append((i, similarity))
        #dot_product = np.dot(texts_np, image_embedding)
        #norm_texts = np.linalg.norm(texts_np, axis=1)
        #norm_image = np.linalg.norm(image_embedding)
        #cosine_scores = dot_product / (norm_texts * norm_image)
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i, score in similarities[:5]:
            movie = self.movies[i]
            truncated_score = math.floor(score * 1000) / 1000
            results.append({
                "id": i + 1,
                "title": movie['title'],
                "description": movie['description'][:100],
                "score": truncated_score,
            })
        return results


def verify_image_embedding(image_path):
    searcher = MultimodalSearch()
    embedding = searcher.embed_image(image_path)
    print(type(embedding), embedding.shape)
    return embedding.shape

def image_search_command(image_path):
    searcher = MultimodalSearch()
    results = searcher.search_with_image(image_path)
    return results

