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

# Contains track_name, artist_name and album_name
combined_dedup = data.drop_duplicates(subset=['track_name', 'artist_name', 'album_name'], keep='first').reset_index(drop=True)
print(f"Original number of tracks: {len(data)}")
print(f"Deduplicated number of combined: {len(combined_dedup)}")

# Prepare corpus with character n-grams
corpus = []
for idx, row in combined_dedup.iterrows():
    if idx % 1000 == 0:
        print(f"Processing combined {idx}/{len(combined_dedup)}")
    track_name = str(row['track_name']) if pd.notna(row['track_name']) else ''
    artist_name = str(row['artist_name']) if pd.notna(row['artist_name']) else ''
    album_name = str(row['album_name']) if pd.notna(row['album_name']) else ''
    combined_text = f"{track_name} {artist_name} {album_name}"
    corpus.append(tokenize(combined_text, config['bm25']['ngram_range']))
print("Corpus: ", corpus[0:30])

# Create sparse vector embeddings using BM25
bm25_combined = BM25Okapi(corpus)

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
    pickle.dump(bm25_combined, f)
print(f"Saved BM25 model to {config['output']['bm25_model_path']}")

with open(config['output']['vocab_path'], 'wb') as f:
    pickle.dump(vocab, f)
print(f"Saved vocabulary to {config['output']['vocab_path']}")