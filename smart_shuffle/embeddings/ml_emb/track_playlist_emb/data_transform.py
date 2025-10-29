# /home/romain/Documents/Personal/Projects/Soundtrack_ML/soundtrack-ml/w2v/data/transform_data.py

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pickle
from collections import Counter
import logging
import yaml

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class DataTransformer:
    def __init__(self, raw_data_path: str, processed_data_path: str = "processed"):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.processed_data_path.mkdir(exist_ok=True)
        
    def load_playlist_data(self, max_files: int = None) -> List[Dict]:
        """Load Spotify Million Playlist Dataset from JSON files"""
        try:
            json_files = list(self.raw_data_path.glob("mpd.slice.*.json"))
            
            if not json_files:
                raise FileNotFoundError("No JSON files found in the raw data directory")
            
            if max_files:
                json_files = json_files[:max_files]
            
            print(f"Loading {len(json_files)} JSON files...")
            
            all_playlists = []
            for i, json_file in enumerate(json_files):
                if i % 10 == 0:
                    print(f"Processing file {i+1}/{len(json_files)}: {json_file.name}")
                
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_playlists.extend(data['playlists'])
            
            print(f"Loaded {len(all_playlists)} playlists from {len(json_files)} files")
            return all_playlists
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return []
    
    def extract_track_info(self, playlists: List[Dict]) -> pd.DataFrame:
        """Extract track information from playlists"""
        tracks_data = []
        
        for playlist in playlists:
            playlist_id = playlist.get('pid', '')
            playlist_name = playlist.get('name', '')
            
            for track in playlist.get('tracks', []):
                track_info = {
                    'playlist_id': playlist_id,
                    'playlist_name': playlist_name,
                    'track_name': track.get('track_name', ''),
                    'artist_name': track.get('artist_name', ''),
                    'album_name': track.get('album_name', ''),
                    'duration_ms': track.get('duration_ms', 0),
                    'pos': track.get('pos', 0)
                }
                tracks_data.append(track_info)
        
        df = pd.DataFrame(tracks_data)
        print(f"Extracted {len(df)} track entries")
        return df
    
    def create_training_sentences(self, df: pd.DataFrame) -> List[List[str]]:
        """Create sentences for Word2Vec training from track data"""
        sentences = []
        
        # Group by playlist to create playlist-level sequences
        for playlist_id, group in df.groupby('playlist_id'):
            playlist_sentence = []
            
            # Create combined track_artist_album tokens
            for idx, row in group.iterrows():
                track_name = row['track_name']
                artist_name = row['artist_name']
                album_name = row['album_name']
                
                # Build combined token parts
                token_parts = []
                
                if isinstance(track_name, str) and track_name.strip():
                    track_tokens = self._tokenize_text(track_name)
                    token_parts.extend(track_tokens)
                
                if isinstance(artist_name, str) and artist_name.strip():
                    artist_tokens = self._tokenize_text(artist_name)
                    token_parts.extend(artist_tokens)
                
                if isinstance(album_name, str) and album_name.strip():
                    album_tokens = self._tokenize_text(album_name)
                    token_parts.extend(album_tokens)
                
                # Join all parts with underscore to create single token
                if token_parts:
                    combined_token = '_'.join(token_parts)
                    playlist_sentence.append(combined_token)
            
            if len(playlist_sentence) > 1:
                sentences.append(playlist_sentence)
        
        print(f"Created {len(sentences)} training sentences")
        return sentences
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for Word2Vec training"""
        if not isinstance(text, str):
            return []
        
        # Convert to lowercase and split
        tokens = text.lower().split()
        
        # add logic if needed to filter tokens
        # Filter tokens: keep only alphanumeric, length > 1
        # tokens = [token for token in tokens if len(token) > 1 and token.isalnum()]
        
        return tokens
    
    def get_vocabulary_stats(self, sentences: List[List[str]]) -> Dict[str, Any]:
        """Get vocabulary statistics"""
        all_tokens = []
        for sentence in sentences:
            all_tokens.extend(sentence)
        
        token_counts = Counter(all_tokens)
        
        stats = {
            'total_tokens': len(all_tokens),
            'unique_tokens': len(token_counts),
            'total_sentences': len(sentences),
            'avg_sentence_length': np.mean([len(s) for s in sentences]),
            'most_common_tokens': token_counts.most_common(20)
        }
        
        return stats
    
    def save_processed_data(self, sentences: List[List[str]], filename: str = "training_sentences"):
        """Save processed sentences for training"""
        # Save as pickle for fast loading
        pickle_path = self.processed_data_path / f"{filename}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(sentences, f)
        print(f"Sentences saved to: {pickle_path}")
        
        # Save as text file for inspection
        txt_path = self.processed_data_path / f"{filename}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            for sentence in sentences[:1000]:  # Save first 1000 sentences for inspection
                f.write(' '.join(sentence) + '\n')
        print(f"Sample sentences saved to: {txt_path}")
        
        return pickle_path, txt_path
    
    def save_dataframe(self, df: pd.DataFrame, filename: str = "tracks_data"):
        """Save processed dataframe"""
        csv_path = self.processed_data_path / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        print(f"DataFrame saved to: {csv_path}")
        return csv_path
    
    def process_all_data(self, max_files: int = 10):
        """Complete data processing pipeline"""
        print("Starting data transformation pipeline...")
        
        # Load data
        playlists = self.load_playlist_data(max_files=max_files)
        if not playlists:
            print("No data loaded. Exiting.")
            return
        
        # Extract track information
        df = self.extract_track_info(playlists)
        self.save_dataframe(df, "tracks_data")
        
        # Create different types of training sentences
        print("\nCreating playlist-based sentences...")
        playlist_sentences = self.create_training_sentences(df)
        self.save_processed_data(playlist_sentences, "playlist_sentences")
        
        # Get and save statistics
        print("\nGenerating vocabulary statistics...")
        playlist_stats = self.get_vocabulary_stats(playlist_sentences)
        
        stats = {
            'playlist_sentences': playlist_stats,
        }
        
        stats_path = self.processed_data_path / "vocabulary_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Vocabulary statistics saved to: {stats_path}")
        
        print("\nData transformation completed!")
        return stats

def main():
    # Load configuration from YAML file
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get paths from config
    RAW_DATA_PATH = config['data']['raw_data_path']
    PROCESSED_DATA_PATH = config['data']['processed_data_path']
    
    # Initialize transformer
    transformer = DataTransformer(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    
    # Process data (start with 10 files for testing)
    stats = transformer.process_all_data(max_files=config['data_transform']['max_files'])
    
    if stats:
        print("\nVocabulary Statistics Summary:")
        for sentence_type, stats_data in stats.items():
            print(f"\n{sentence_type}:")
            print(f"  Total sentences: {stats_data['total_sentences']}")
            print(f"  Unique tokens: {stats_data['unique_tokens']}")
            print(f"  Average sentence length: {stats_data['avg_sentence_length']:.2f}")
            print(f"  Most common tokens: {[token for token, count in stats_data['most_common_tokens'][:5]]}")

if __name__ == "__main__":
    main()