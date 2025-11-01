# retrievers/base_retriever.py - Base class
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from qdrant_client import QdrantClient
from qdrant_client import models
import numpy as np
from collections import Counter

class BaseRetriever(ABC):
    """Base class for all retrievers"""

    def __init__(self, name: str, description: str, parameters: Dict[str, Any], qdrant_client: QdrantClient, qdrant_collection_name: str):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.qdrant_client = qdrant_client
        self.qdrant_collection_name = qdrant_collection_name

    def tokenize(self, text: str, ngram_range: Tuple[int, int]) -> List[str]:
        tokens = []
        for n_gram_size in range(ngram_range[0], ngram_range[1]):
            tokens.extend([text[i:i+n_gram_size] for i in range(len(text)-n_gram_size+1)])
        return tokens

    # Convert BM25 scores to Qdrant SparseVector format
    def convert_bm25_to_qdrant_sparse(self, doc_tokens, vocab):
        """Convert BM25 document to Qdrant SparseVector format"""
        token_counts = Counter(doc_tokens)
        
        indices = []
        values = []
        for token, count in token_counts.items():
            if token in vocab:
                indices.append(np.uint32(vocab[token]))
                values.append(np.float32(count))
        
        return models.SparseVector(
            indices=[np.uint32(i) for i in indices],
            values=[np.float32(v) for v in values]
        )

    # TODO: this is not good practice, we want to later set this as abstract method and implement it in the child classes, all child classes NEED it
    def search_qdrant(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Execute search query.
        
        Args:
            query: Search query string
            top_k: Number of results to return
        
        Returns:
            List of search results with standardized format
        """
        # Search in Qdrant
        search_results = self.qdrant_client.query_points(
            collection_name=self.qdrant_collection_name,
            query=query,
            limit=top_k,
            using="bm25",
            with_payload=True
        ).points
        
        return search_results