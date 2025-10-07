import numpy as np
from embeddings import cosine_similarity

class SemanticCache:
    def __init__(self, embedding_model, similarity_threshold=0.9):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.cache = {}  # {session_id: [ {embedding, query, response}, ... ]}

    def get_cached_response(self, session_id, query_with_context):
        """Return cached response if similar query exists"""
        if session_id not in self.cache:
            return None

        query_vec = self.embedding_model.embed_text(query_with_context)
        for entry in self.cache[session_id]:
            sim = cosine_similarity(query_vec, entry["embedding"])
            if sim >= self.similarity_threshold:
                return entry["response"]

        return None

    def store_response(self, session_id, query_with_context, response):
        """Store new embedding + response in cache"""
        query_vec = self.embedding_model.embed_text(query_with_context)
        if session_id not in self.cache:
            self.cache[session_id] = []
        self.cache[session_id].append({
            "embedding": query_vec,
            "query": query_with_context,
            "response": response
        })
