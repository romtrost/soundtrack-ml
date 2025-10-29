# /home/romain/Documents/Personal/Projects/Soundtrack_ML/soundtrack-ml/w2v/training/train_w2v.py

import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import logging
from pathlib import Path
import pickle
import json
from typing import List, Dict, Any
import yaml

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class EpochLogger(CallbackAny2Vec):
    """Callback to log information about training"""
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        print(f"Epoch #{self.epoch} end")
        self.epoch += 1

class Word2VecTrainer:
    def __init__(self, data_path: str, model_save_path: str = "models"):
        self.data_path = data_path
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(exist_ok=True)
        self.sentences = []
    
    def load_sentences(self) -> List[List[str]]:
        """Load sentences from a pickle file"""
        with open(self.data_path, 'rb') as f:
            sentences = pickle.load(f)
        
        print(f"Loaded {len(sentences)} sentences from pickle file")
        return sentences
    
    def train_model(self, 
                   sentences: List[List[str]], 
                   vector_size: int = 100,
                   window: int = 5,
                   min_count: int = 5,
                   workers: int = 4,
                   epochs: int = 5,
                   sg: int = 0) -> Word2Vec:
        """Train Word2Vec model"""
        
        print(f"Training Word2Vec model with {len(sentences)} sentences...")
        print(f"Parameters: vector_size={vector_size}, window={window}, min_count={min_count}")
        
        # Initialize the model
        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=sg,  # 0 for CBOW, 1 for Skip-gram
            callbacks=[EpochLogger()]
        )
        
        # Train the model
        model.train(sentences, total_examples=len(sentences), epochs=epochs)
        
        return model
    
    def save_model(self, model: Word2Vec, model_name: str = "w2v_model"):
        """Save the trained model and metadata"""
        model_path = self.model_save_path / f"{model_name}.model"
        metadata_path = self.model_save_path / f"{model_name}_metadata.json"
        
        # Save the model
        model.save(str(model_path))
        print(f"Model saved to: {model_path}")
        
        # Save metadata
        metadata = {
            "vocab_size": len(model.wv),
            "vector_size": model.wv.vector_size,
            "window": model.window,
            "min_count": model.min_count,
            "epochs": model.epochs,
            "sg": model.sg
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")
    
    def evaluate_model(self, model: Word2Vec, test_words: List[str] = None):
        """Basic evaluation of the trained model"""
        if test_words is None:
            test_words = list(model.wv.key_to_index.keys())[:10]
        
        print(f"\nModel evaluation:")
        print(f"Vocabulary size: {len(model.wv)}")
        print(f"Vector dimensions: {model.wv.vector_size}")
        
        print(f"\nSample words and their vectors:")
        for word in test_words:
            if word in model.wv:
                print(f"'{word}': {model.wv[word]}...")  # Show first 5 dimensions
        
        # Find most similar words for a few test words
        print(f"\nMost similar words:")
        for word in test_words:
            if word in model.wv:
                similar = model.wv.most_similar(word, topn=5)
                print(f"Words similar to '{word}': {[w for w, score in similar]}")

def main():
    # Load configuration from YAML file
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get paths from config
    DATA_PATH = config['data']['ml_train_data_path']
    MODEL_SAVE_PATH = config['data']['model_save_path']
    
    # Initialize trainer
    trainer = Word2VecTrainer(DATA_PATH, MODEL_SAVE_PATH)
    
    print("Preprocessing text data...")
    sentences = trainer.load_sentences()
    
    if len(sentences) < 100:
        print(f"Warning: Only {len(sentences)} sentences found. Consider using more data.")
    
    # Train model
    print("Training Word2Vec model...")
    model = trainer.train_model(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=5,
        workers=4,
        epochs=5,
        sg=0  # CBOW
    )
    
    # Save model
    print("Saving model...")
    trainer.save_model(model, "spotify_w2v")
    
    # Evaluate model
    trainer.evaluate_model(model)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()