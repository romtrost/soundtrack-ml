# Smart-Shuffle project example

The structure is below

(Note: the embeddings/ and reranker/ can be branched out into their own repositories.)

```
smart_shuffle/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── data_fetching/                        # Used to fetch raw data from database, will often involve sql queries
│   ├── fetch_playlist_data.py
│   ├── fetch_track_metadata_data.py
│   └── fetch_track_audio_data.py
├── embeddings/
│   ├── embeddings.py                     # Main embeddings orchestrator
│   ├── custom_emb/
│   │   ├── popularity_emb/
│   │   │   ├── create_popularity_emb.py
│   │   └── date_emb/
│   │       └── create_date_emb.py
│   ├── ml_emb/
│   │   ├── track_playlist_emb/
│   │   │   ├── data/
│   │   │   │   ├── train.csv
│   │   │   │   └── test.csv
│   │   │   ├── data_transform.py
│   │   │   ├── models/
│   │   │   ├── train_w2v.py
│   │   │   ├── evaluate_w2v.py
│   │   │   └── config.yaml
│   │   ├── track_audio_emb/
│   │   │   ├── data/
│   │   │   │   ├── train.csv
│   │   │   │   └── test.csv
│   │   │   ├── data_transform.py
│   │   │   ├── models/
│   │   │   ├── train_cnn.py
│   │   │   ├── evaluate_cnn.py
│   │   │   └── config.yaml
│   │   ├── artist_emb/
│   │   │   ├── data/
│   │   │   │   ├── train.csv
│   │   │   │   └── test.csv
│   │   │   ├── data_transform.py
│   │   │   ├── models/
│   │   │   ├── train_w2v.py
│   │   │   ├── evaluate_w2v.py
│   │   │   ├── upload_w2v.py
│   │   │   ├── upsert_emb.py
│   │   │   └── config.yaml        # whenever you want to train a new model, update tings here like train dates, model_name, etc
│   │   └── genre_emb/
│   │       ├── data/
│   │       │   ├── train.csv
│   │       │   └── test.csv
│   │       ├── data_transform.py
│   │       ├── models/
│   │       ├── train_w2v.py
│   │       ├── evaluate_w2v.py
│   │       └── config.yaml
│   └── evaluation/                       # For system level evaluation (all embeddings combined)
│       ├── embedding_evaluator.py
│       └── system_evaluator.py
├── reranker/
│   ├── data/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── data_transform.py
│   ├── models/
│   ├── train_tt_model.py
│   ├── evaluate_tt_model.py
│   └── config.yaml
├── app/                                  # Include main app here, will receive a request and handle or the serving logic (i.e do ann search on embeddings, fusion, rerank, filter)
│   ├── smart_shuffle_app.py
│   ├── inference/
│   └── docker/
├── deployment/
│   └── task_definitions/
│   └── docker/
├── config/
│   ├── base.yaml
│   └── embeddings.yaml
├── utils/
│   ├── logging.py
│   └── gcp_handler.py
├── tests/
├── notebooks/
└── README.md
```

## Embeddings:

 Track embeddings:
 - track vector 1 (trained w2v, playlist co-occurence)
 - track vector 2 (audio feature vector, trained with CNN)
 - artist vector (trained w2v, playlist co-occurence)
 - genre vector (trained w2v, playlist co-occurence)
 - popularity vector (custom mapping)
 - date vector (custom mapping)

 Dim reduction?
