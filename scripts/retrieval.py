import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from baseline import AUDIO_FEATURES
from mood_parser import mood_to_vector

def load_artifacts(
    encoder_path="outputs/encoder.keras",
    auto_knn_path="outputs/autoencoder_knn.pkl",
    baseline_knn_path="outputs/baseline_knn.pkl",
    scaler_path="outputs/scaler.pkl",
    embeddings_path="outputs/embeddings.npy",
    cluster_path="outputs/kmeans.pkl",
    df_path="outputs/cleaned_data.csv"
):
    encoder = load_model(encoder_path)
    auto_knn = joblib.load(auto_knn_path)
    baseline_knn = joblib.load(baseline_knn_path)
    scaler = joblib.load(scaler_path)
    embeddings = np.load(embeddings_path)
    cluster_model = joblib.load(cluster_path)
    df = pd.read_csv(df_path)
    print("Artifacts loaded successfully.")
    return df, encoder, auto_knn, baseline_knn, scaler, embeddings, cluster_model

# Utility: add normalized match score
def add_match_score(recs, distances):
    recs = recs.copy()
    max_d = max(distances)
    recs["match_score"] = 1 - (distances / max_d)  # closer = higher score
    return recs

def recommend_track_autoencoder(track_name, df, encoder, knn, n_neighbors=10):
    track = df[df["track_name"].str.lower() == track_name.lower()]
    if track.empty:
        print("Track not found.")
        return None
    idx = track.index[0]
    X = df[AUDIO_FEATURES].values
    embedding = encoder.predict(X[idx].reshape(1, -1))
    distances, indices = knn.kneighbors(embedding, n_neighbors=n_neighbors)
    recs = df.iloc[indices[0]]
    recs = add_match_score(recs, distances[0])
    print(f"\nAutoencoder recommendations for '{track_name}':")
    print(recs[["track_name", "artists", "track_genre", "popularity", "match_score"]])
    return recs

def recommend_track_baseline(track_name, df, knn, n_neighbors=10):
    track = df[df["track_name"].str.lower() == track_name.lower()]
    if track.empty:
        print("Track not found.")
        return None
    idx = track.index[0]
    X = df[AUDIO_FEATURES].values
    distances, indices = knn.kneighbors([X[idx]], n_neighbors=n_neighbors)
    recs = df.iloc[indices[0]]
    recs = add_match_score(recs, distances[0])
    print(f"\nBaseline recommendations for '{track_name}':")
    print(recs[["track_name", "artists", "track_genre", "popularity", "match_score"]])
    return recs

def recommend_within_cluster(track_name, df, n_neighbors=10):
    track = df[df["track_name"].str.lower() == track_name.lower()]
    if track.empty:
        print("Track not found.")
        return None
    cluster = track["cluster"].values[0]  # use precomputed cluster
    same_cluster = df[df["cluster"] == cluster]
    recs = same_cluster.head(n_neighbors)
    print(f"\nRecommendations for '{track_name}' within cluster {cluster}:")
    print(recs[["track_name", "artists", "track_genre", "popularity"]])
    return recs

def recommend_mood_autoencoder(mood, df, encoder, knn, scaler, n_neighbors=10, genre=None):
    vec = mood_to_vector(mood, scaler)
    if vec is None:
        return None
    embedding = encoder.predict(vec)
    distances, indices = knn.kneighbors(embedding, n_neighbors=n_neighbors)
    recs = df.iloc[indices[0]]
    recs = add_match_score(recs, distances[0])
    if genre:
        recs = recs[recs["track_genre"].str.lower() == genre.lower()]
    print(f"\nAutoencoder recommendations for mood '{mood}'" + (f" in genre '{genre}'" if genre else "") + ":")
    print(recs[["track_name", "artists", "track_genre", "popularity", "match_score"]])
    return recs

if __name__ == "__main__":
    df, encoder, auto_knn, baseline_knn, scaler, embeddings, cluster_model = load_artifacts()
    recommend_track_autoencoder("Shape of You", df, encoder, auto_knn)
    recommend_track_baseline("Shape of You", df, baseline_knn)
    recommend_within_cluster("Shape of You", df)
    recommend_mood_autoencoder("happy", df, encoder, auto_knn, scaler, genre="pop")
