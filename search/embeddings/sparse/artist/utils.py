import yaml
import numpy as np
from collections import Counter
from qdrant_client import models

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def tokenize(text, ngram_range):
    tokens = []
    for n_gram_size in range(ngram_range[0], ngram_range[1]):
        tokens.extend([text[i:i+n_gram_size] for i in range(len(text)-n_gram_size+1)])
    return tokens

# Convert BM25 scores to Qdrant SparseVector format
def convert_bm25_to_qdrant_sparse(doc_tokens, vocab):
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
