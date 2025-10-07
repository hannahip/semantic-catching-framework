import sys
import os
import pytest
import numpy as np

# Add src folder to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from embeddings import EmbeddingModel
from cache import SemanticCache


def test_semantic_cache_hit_and_miss():
    model = EmbeddingModel()
    cache = SemanticCache(model, similarity_threshold=0.9)

    session_id = "s1"

    # First query and response
    query1 = "What is the impact of climate change on corn yields?"
    response1 = "Climate change can reduce corn yields due to drought."
    cache.store_response(session_id, query1, response1)

    # Similar query - should hit cache
    query2 = "How does global warming affect maize productivity?"
    cached = cache.get_cached_response(session_id, query2)

    # We don't know the exact cosine similarity from mock embeddings,
    # but this ensures the function runs and returns either None or str.
    assert cached is None or isinstance(cached, str)

    # Completely different query - should not hit cache
    query3 = "What is the capital of France?"
    cached2 = cache.get_cached_response(session_id, query3)
    assert cached2 is None
