from typing import List

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """WRAPPER FOR SENTENCE TRANSFORMER EMBEDDINGS. **"""

    def __init__(self, model_name: str, normalize_embeddings: bool = True):
        """INITIALIZE EMBEDDING MODEL. **"""
        self.model = SentenceTransformer(model_name)
        self.normalize_embeddings = normalize_embeddings

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """EMBED A LIST OF TEXTS. **"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """EMBED A SINGLE QUERY. **"""
        embedding = self.model.encode(
            query,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return embedding.tolist()