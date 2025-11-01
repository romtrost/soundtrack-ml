from qdrant_client import QdrantClient
import yaml
from gensim.models import Word2Vec
import time


def similarity_search(qdrant_client: QdrantClient, model: Word2Vec, track_name: str, collection_name: str, top_k: int = 10):
    """Search for similar tracks using the Word2Vec model and Qdrant"""
    
    # Check if track exists in model vocabulary
    if track_name not in model.wv:
        print(f"Track '{track_name}' not found in model vocabulary")
        return []
    
    # Get the embedding for the query track
    query_vector = model.wv[track_name].tolist()
    
    print(f"Searching for tracks similar to '{track_name}'...")
    
    # Search in Qdrant
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=("dense_vector", query_vector),
        limit=top_k + 1  # +1 because the query track itself might be in results
    )
    
    # Filter out the query track itself and format results
    results = []
    for hit in search_results:
        similar_track = hit.payload.get("track_name")
        if similar_track != track_name:  # Exclude the query track
            results.append({
                "track_name": similar_track,
                "similarity_score": hit.score
            })
    
    return results[:top_k]


if __name__ == "__main__":

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    QDRANT_URL = config['qdrant']['url']
    QDRANT_API_KEY = config['qdrant']['api_key']
    QDRANT_COLLECTION_NAME = config['qdrant']['collection_name']
    MODEL_NAME = config['data']['model_name']
    MODEL_SAVE_PATH = config['data']['model_save_path']

    # Initialize Qdrant client
    qdrant_client = QdrantClient(
        QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=30.0
    )

    # Load the Word2Vec model
    model_path = MODEL_SAVE_PATH + "/" + MODEL_NAME
    print(f"Loading Word2Vec model from {model_path}...")
    model = Word2Vec.load(model_path)

    # Example: Search for similar tracks
    query_track = "lost_but_won_hans_zimmer_rush"  # Replace with actual track name
    top_k = 30
    
    start_time = time.time()
    results = similarity_search(qdrant_client, model, query_track, QDRANT_COLLECTION_NAME, top_k)
    end_time = time.time()
    print(f"Search completed in {end_time - start_time:.4f} seconds")
    
    print(f"\nTop {len(results)} tracks similar to '{query_track}':")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['track_name']} (similarity: {result['similarity_score']:.4f})")