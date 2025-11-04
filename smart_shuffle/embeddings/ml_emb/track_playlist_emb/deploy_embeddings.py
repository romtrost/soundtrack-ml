from qdrant_client import QdrantClient
import yaml
from qdrant_client.http.models import PointStruct
from qdrant_client import models
from gensim.models import Word2Vec
import uuid


# Create collection if it doesn't exist
def create_collection(qdrant_client: QdrantClient, collection_name: str, vector_size: int):
    # Only create collection if it doesn't exist
    if not qdrant_client.collection_exists(collection_name):
        print(f"Creating collection '{collection_name}'...")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense_vector": models.VectorParams(        #Don't need to have a dict here, only neeeded if want multiple vector names
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            }
        )
    else:
        print(f"Collection '{collection_name}' already exists")

def deploy_embeddings_from_w2v_model(qdrant_client: QdrantClient, model_path: str, collection_name: str, vector_size: int):
    """Load trained Word2Vec model and deploy embeddings to Qdrant"""
    
    # Load the trained Word2Vec model
    print(f"Loading Word2Vec model from {model_path}...")
    model = Word2Vec.load(model_path)
    
    # Extract embeddings and track names from the model vocabulary
    track_names = list(model.wv.index_to_key)
    embeddings = [model.wv[track_name] for track_name in track_names]
    
    print(f"Preparing to upload {len(track_names)} embeddings to Qdrant...")
    
    # Create points for Qdrant
    points = [
        PointStruct(
            id=uuid.uuid4().hex, 
            vector={"dense_vector": embedding.tolist()}, 
            payload={"track_name": track_name}
        ) 
        for i, (embedding, track_name) in enumerate(zip(embeddings, track_names))
    ]
    
    # Upload to Qdrant in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        qdrant_client.upsert(collection_name=collection_name, points=batch)
        print(f"Uploaded batch {i // batch_size + 1}/{(len(points) + batch_size - 1) // batch_size}")
    
    print(f"Successfully deployed {len(track_names)} embeddings to Qdrant collection '{collection_name}'")


if __name__ == "__main__":

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    QDRANT_URL = config['qdrant']['url']
    QDRANT_API_KEY = config['qdrant']['api_key']
    QDRANT_COLLECTION_NAME = config['qdrant']['collection_name']
    MODEL_NAME = config['data']['model_name']
    VECTOR_SIZE = config['model']['vector_size']
    MODEL_SAVE_PATH = config['data']['model_save_path']

    qdrant_client = QdrantClient(
        QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=30.0
    )

    # Get model path from config
    model_path = MODEL_SAVE_PATH + "/" + MODEL_NAME

    create_collection(qdrant_client, QDRANT_COLLECTION_NAME, VECTOR_SIZE)
    deploy_embeddings_from_w2v_model(qdrant_client, model_path, QDRANT_COLLECTION_NAME, VECTOR_SIZE)