import yaml
import pickle
from rank_bm25 import BM25Okapi
from utils import tokenize, load_config, convert_bm25_to_qdrant_sparse
import os
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.http.models import PointStruct

# Load configuration
config = load_config('config.yaml')

# Load data
data = pd.read_csv(config['data']['data_path'])

# Deduplicate tracks based on track_name and artist_name
artists_dedup = data.drop_duplicates(subset=['artist_name'], keep='first').reset_index(drop=True)
print(f"Original number of tracks: {len(data)}")
print(f"Deduplicated number of artists: {len(artists_dedup)}")

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=config['qdrant']['url'],
    api_key=config['qdrant']['api_key'],
    timeout=config['qdrant']['timeout']
)

COLLECTION_NAME = config['qdrant']['collection_name']

qdrant_client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={},  # Empty dict for dense vectors
    sparse_vectors_config={
        "bm25": models.SparseVectorParams(
            index=models.SparseIndexParams(),
            modifier=models.Modifier.IDF  # Enable IDF for proper BM25 scoring
        )
    }
)

# Load corpus
with open(config['output']['corpus_path'], 'rb') as f:
    corpus = pickle.load(f)

# Load vocabulary
with open(config['output']['vocab_path'], 'rb') as f:
    vocab = pickle.load(f)

# Upload embeddings to Qdrant (do this once)
print("Uploading embeddings to Qdrant...")
BATCH_SIZE = config['qdrant']['upsert_batch_size']

for i in range(0, len(artists_dedup), BATCH_SIZE):
    batch_points = []
    for j in range(i, min(i + BATCH_SIZE, len(artists_dedup))):
        sparse_vec = convert_bm25_to_qdrant_sparse(corpus[j], vocab)
        
        batch_points.append(
            PointStruct(
                id=j,
                vector={"bm25": sparse_vec},
                payload={
                    "artist_name": artists_dedup.iloc[j]['artist_name'],
                }
            )
        )
    
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=batch_points
    )
    
    if (i // BATCH_SIZE + 1) % 10 == 0:
        print(f"Uploaded {i + BATCH_SIZE} / {len(artists_dedup)} embeddings...")

print(f"Successfully uploaded {len(artists_dedup)} embeddings to Qdrant!")