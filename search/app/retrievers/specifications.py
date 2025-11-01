# initialise the retrievers specifications here
from qdrant_client import QdrantClient
from retrievers.track_retriever import TrackRetriever
from retrievers.artist_retriever import ArtistRetriever
from retrievers.album_retriever import AlbumRetriever
from retrievers.playlist_retriever import PlaylistRetriever
from retrievers.combined_retriever import CombinedRetriever

from retrievers.utils import load_config
from utils.logging import setup_logger
import time

logger = setup_logger(__name__)

def create_retriever_dict():
    """
    Create a dictionary of retrievers with initialized Qdrant client.
    """
    start_time = time.time()
    logger.info("Creating retriever dictionary...")
    #TODO: pydantic everything
    # Load configuration
    config = load_config('retrievers/config.yaml')

    # Initialize Qdrant client

    # Initialize Qdrant client
    qdrant_client = QdrantClient(
        url=config['qdrant']['url'],
        api_key=config['qdrant']['api_key'],
        timeout=config['qdrant']['timeout']
    )

    # Create retriever dictionary
    retriever_dict = {
        "track": TrackRetriever(
            name="track",
            description="Track retriever",
            parameters={
                "ngram_range": config['bm25']['ngram_range'], 
                "vocab_path": config['bm25']['vocab_path']['track']
                },
            qdrant_client=qdrant_client,
            qdrant_collection_name=config['qdrant']['collections']['track']['sparse']
        ),
        "artist": ArtistRetriever(
            name="artist",
            description="Artist retriever",
            parameters={
                "ngram_range": config['bm25']['ngram_range'], 
                "vocab_path": config['bm25']['vocab_path']['artist']
                },
            qdrant_client=qdrant_client,
            qdrant_collection_name=config['qdrant']['collections']['artist']['sparse']
        ),
        "album": AlbumRetriever(
            name="album",
            description="Album retriever",
            parameters={
                "ngram_range": config['bm25']['ngram_range'], 
                "vocab_path": config['bm25']['vocab_path']['album']
                },
            qdrant_client=qdrant_client,
            qdrant_collection_name=config['qdrant']['collections']['album']['sparse']
        ),
        "playlist": PlaylistRetriever(
            name="playlist",
            description="Playlist retriever",
            parameters={
                "ngram_range": config['bm25']['ngram_range'], 
                "vocab_path": config['bm25']['vocab_path']['playlist']
                },
            qdrant_client=qdrant_client,
            qdrant_collection_name=config['qdrant']['collections']['playlist']['sparse']
        ),
        "combined": CombinedRetriever(
            name="combined",
            description="Combined retriever",
            parameters={
                "ngram_range": config['bm25']['ngram_range'], 
                "vocab_path": config['bm25']['vocab_path']['combined']
                },
            qdrant_client=qdrant_client,
            qdrant_collection_name=config['qdrant']['collections']['combined']['sparse']
        )
    }
    elapsed_time = time.time() - start_time
    logger.info(f"Retriever dictionary created successfully in {elapsed_time:.2f} seconds")
    return retriever_dict