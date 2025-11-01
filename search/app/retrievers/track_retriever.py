from retrievers.base import BaseRetriever
from typing import List, Dict, Any
from qdrant_client import QdrantClient

from retrievers.utils import load_vocab
from utils.logging import setup_logger

logger = setup_logger(__name__)

class TrackRetriever(BaseRetriever):
    """Retriever for track search using BM25 sparse vectors"""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any], qdrant_client: QdrantClient, qdrant_collection_name: str):
        # Could initialize Qdrant client here if needed
        super().__init__(name, description, parameters, qdrant_client, qdrant_collection_name)
        self.vocab = load_vocab(self.parameters['vocab_path']) #TODO: vocab should be stored in cloud storage
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for tracks.
        
        Returns standardized format:
        {
            "track_name": str,
            "artist_name": str,
            "album_name": str | None,
            "score": float,
            "entity_type": "track"
        }
        """
        # Tokenize query
        query_tokens = self.tokenize(query, self.parameters['ngram_range'])
        
        # Convert query to sparse vector
        query_sparse = self.convert_bm25_to_qdrant_sparse(query_tokens, self.vocab)
        
        # Search in Qdrant
        search_results = self.search_qdrant(query_sparse, top_k=top_k)

        # Format results
        results = []
        for hit in search_results:
            results.append({
                "track_name": hit.payload["track_name"],
                "artist_name": hit.payload["artist_name"],
                "score": hit.score
            })

        return results
        """
        # Standardize format
        standardized = []
        for result in results:
            standardized.append({
                "track_name": result.get("track_name"),
                "artist_name": result.get("artist_name"),
                "album_name": result.get("album_name"),
                "score": result.get("score", 0.0),
                "entity_type": "track"
            })
        
        return standardized
        """

