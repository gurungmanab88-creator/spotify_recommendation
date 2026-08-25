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
    # Load encoder
    encoder = load_model(encoder_path)

    # Load autoencoder KNN
    auto_knn = joblib.load(auto_knn_path)

    # Load baseline KNN
    baseline_knn = joblib.load(baseline_knn_path)

    # Load scaler
    scaler = joblib.load(scaler_path)

    # Load embeddings
    embeddings = np.load(embeddings_path)

    # Load cluster model
    cluster_model = joblib.load(cluster_path)

    # Load dataframe
    df = pd.read_csv(df_path)

    print("Artifacts loaded successfully.")
    return df, encoder, auto_knn, baseline_knn, scaler, embeddings, cluster_model


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
    print(f"\nAutoencoder recommendations for '{track_name}':")
    print(recs[["track_name", "artists", "track_genre", "popularity"]])
    return recs

def recommend_within_cluster(track_name, df, encoder, knn, cluster_model, n_neighbors=10):
    track = df[df["track_name"].str.lower() == track_name.lower()]
    if track.empty:
        print("Track not found.")
        return None
    idx = track.index[0]
    X = df[AUDIO_FEATURES].values
    embedding = encoder.predict(X[idx].reshape(1, -1))
    cluster = cluster_model.predict(embedding)[0]
    cluster_indices = cluster_model.predict(encoder.predict(X))
    same_cluster = df[cluster_indices == cluster]
    print(f"\nRecommendations for '{track_name}' within cluster {cluster}:")
    print(same_cluster[["track_name", "artists", "track_genre", "popularity"]].head(n_neighbors))
    return same_cluster



def recommend_track_baseline(track_name, df, knn, n_neighbors=10):
    track = df[df["track_name"].str.lower() == track_name.lower()]
    if track.empty:
        print("Track not found.")
        return None

    idx = track.index[0]
    X = df[AUDIO_FEATURES].values
    distances, indices = knn.kneighbors([X[idx]], n_neighbors=n_neighbors)
    recs = df.iloc[indices[0]]
    print(f"\nBaseline recommendations for '{track_name}':")
    print(recs[["track_name", "artists", "track_genre", "popularity"]])
    return recs


def recommend_mood_autoencoder(mood, df, encoder, knn, n_neighbors=10, genre=None):
    vec = mood_to_vector(mood)
    if vec is None:
        return None

    embedding = encoder.predict(vec)
    distances, indices = knn.kneighbors(embedding, n_neighbors=n_neighbors)
    recs = df.iloc[indices[0]]

    if genre:
        recs = recs[recs["track_genre"].str.lower() == genre.lower()]

    print(f"\nAutoencoder recommendations for mood '{mood}'" + (f" in genre '{genre}'" if genre else "") + ":")
    print(recs[["track_name", "artists", "track_genre", "popularity"]])
    return recs


if __name__ == "__main__":
    df, encoder, auto_knn, baseline_knn, scaler, embeddings, cluster_model = load_artifacts()
    recommend_track_autoencoder("Shape of You", df, encoder, auto_knn)
    recommend_track_baseline("Shape of You", df, baseline_knn)
    recommend_mood_autoencoder("happy", df, encoder, auto_knn, genre="pop")
