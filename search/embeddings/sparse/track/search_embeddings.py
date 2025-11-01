from utils import tokenize, load_config, convert_bm25_to_qdrant_sparse
from qdrant_client import QdrantClient
import pickle

# Search function
def search_tracks(query, top_k=10):
    """Search for tracks using BM25 sparse vectors"""
    # Tokenize query
    query_tokens = tokenize(query, config['bm25']['ngram_range'])
    
    # Convert query to sparse vector
    query_sparse = convert_bm25_to_qdrant_sparse(query_tokens, vocab)
    
    # Search in Qdrant
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_sparse,
        limit=top_k,
        using="bm25",
        with_payload=True
    ).points
    
    # Format results
    results = []
    for hit in search_results:
        results.append({
            "track_name": hit.payload["track_name"],
            "artist_name": hit.payload["artist_name"],
            "score": hit.score
        })
    
    return results

# Load configuration
config = load_config('config.yaml')

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=config['qdrant']['url'],
    api_key=config['qdrant']['api_key'],
    timeout=config['qdrant']['timeout']
)

COLLECTION_NAME = config['qdrant']['collection_name']

# Load vocabulary
with open(config['output']['vocab_path'], 'rb') as f:
    vocab = pickle.load(f)

# Test search
query = "Momentum Don Diablo"
print(f"\nSearching for: '{query}'")
results = search_tracks(query, top_k=10)
print(f"\nSearch results for: '{query}'")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['track_name']} - {result['artist_name']} (score: {result['score']:.4f})")
