import pandas as pd
import yaml
import pickle
from rank_bm25 import BM25Okapi
from utils import tokenize, load_config
import os

# Load configuration
config = load_config('config.yaml')

# Load data
data = pd.read_csv(config['data']['data_path'])

# Deduplicate tracks based on track_name and artist_name
playlists_dedup = data.drop_duplicates(subset=['playlist_name'], keep='first').reset_index(drop=True)
print(f"Original number of artists: {len(data)}")
print(f"Deduplicated number of playlists: {len(playlists_dedup)}")

# Prepare corpus with character n-grams
corpus = []
for idx, row in playlists_dedup.iterrows():
    if idx % 1000 == 0:
        print(f"Processing playlist {idx}/{len(playlists_dedup)}")
    playlist_name = str(row['playlist_name']) if pd.notna(row['playlist_name']) else ''
    corpus.append(tokenize(playlist_name, config['bm25']['ngram_range']))
print("Corpus: ", corpus[0:30])

# Create sparse vector embeddings using BM25
bm25_playlist = BM25Okapi(corpus)

print(f"Created BM25 index")
print(f"Number of documents in corpus: {len(corpus)}")

# Build vocabulary from corpus
vocab = {}
vocab_idx = 0
for doc in corpus:
    for token in doc:
        if token not in vocab:
            vocab[token] = vocab_idx
            vocab_idx += 1

print(f"Built vocabulary with {len(vocab)} unique tokens")


if not os.path.exists(config['output']['output_path']):
    os.makedirs(config['output']['output_path'])
# Save corpus, model and vocab
with open(config['output']['corpus_path'], 'wb') as f:
    pickle.dump(corpus, f)
print(f"Saved corpus to {config['output']['corpus_path']}")

with open(config['output']['bm25_model_path'], 'wb') as f:
    pickle.dump(bm25_playlist, f)
print(f"Saved BM25 model to {config['output']['bm25_model_path']}")

with open(config['output']['vocab_path'], 'wb') as f:
    pickle.dump(vocab, f)
print(f"Saved vocabulary to {config['output']['vocab_path']}")