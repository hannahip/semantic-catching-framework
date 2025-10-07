import numpy as np

class EmbeddingModel:
    def __init__(self, dim=8):
        # dim is the embedding size for this mock
        self.dim = dim

    def embed_text(self, text: str):
        # For now, create a deterministic mock embedding
        np.random.seed(abs(hash(text)) % (2**32))
        return np.random.rand(self.dim)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

