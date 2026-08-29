import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from config import AUDIO_FEATURES
from mood_parser import mood_to_vector

# --- Metric Functions ---

def evaluate_genre_consistency(df, encoder, knn, track_name, n_neighbors=10, use_encoder=False, scaler=None):
    track = df[df["track_name"].str.lower() == track_name.lower()]
    if track.empty:
        print("Track not found.")
        return None

    genre = track["track_genre"].values[0]
    idx = track.index[0]
    X = df[AUDIO_FEATURES].values

    if use_encoder:
        query = encoder.predict(X[idx].reshape(1, -1))
    else:
        query = scaler.transform(X[idx].reshape(1, -1))

    distances, indices = knn.kneighbors(query, n_neighbors=n_neighbors)
    recs = df.iloc[indices[0]]

    same_genre = (recs["track_genre"].str.lower() == genre.lower()).sum()
    consistency = same_genre / n_neighbors
    print(f"Genre consistency for '{track_name}' ({genre}): {consistency:.2f}")
    return consistency


def evaluate_mood_alignment(df, encoder, knn, mood, n_neighbors=10, use_encoder=False, scaler=None):
    vec = mood_to_vector(mood)
    if vec is None:
        print("Mood not recognized.")
        return None

    if use_encoder:
        query = encoder.predict(vec)
    else:
        query = vec

    distances, indices = knn.kneighbors(query, n_neighbors=n_neighbors)
    recs = df.iloc[indices[0]]

    avg_features = recs[AUDIO_FEATURES].mean().values
    similarity = np.dot(vec.flatten(), avg_features) / (
        np.linalg.norm(vec.flatten()) * np.linalg.norm(avg_features)
    )
    print(f"Mood alignment for '{mood}': {similarity:.2f}")
    return similarity


def evaluate_diversity(df, encoder, knn, track_name, n_neighbors=10, use_encoder=False, scaler=None):
    track = df[df["track_name"].str.lower() == track_name.lower()]
    if track.empty:
        print("Track not found.")
        return None

    idx = track.index[0]
    X = df[AUDIO_FEATURES].values

    if use_encoder:
        query = encoder.predict(X[idx].reshape(1, -1))
    else:
        query = scaler.transform(X[idx].reshape(1, -1))

    distances, indices = knn.kneighbors(query, n_neighbors=n_neighbors)
    recs = df.iloc[indices[0]]

    if use_encoder:
        rec_embeddings = encoder.predict(recs[AUDIO_FEATURES].values)
        sim_matrix = np.dot(rec_embeddings, rec_embeddings.T)
        norms = np.linalg.norm(rec_embeddings, axis=1)
        cosine_sim = sim_matrix / np.outer(norms, norms)
        mask = ~np.eye(cosine_sim.shape[0], dtype=bool)
        diversity = 1 - cosine_sim[mask].mean()
    else:
        diversity = recs["track_genre"].nunique() / n_neighbors

    print(f"Diversity for '{track_name}': {diversity:.2f}")
    return diversity


# --- Artifact Loader ---

def load_artifacts(
    encoder_path="outputs/encoder.keras",
    auto_knn_path="outputs/autoencoder_knn.pkl",
    baseline_knn_path="outputs/baseline_knn.pkl",
    scaler_path="outputs/scaler.pkl",
    embeddings_path="outputs/embeddings.npy",
    df_path="outputs/cleaned_data.csv"
):
    encoder = load_model(encoder_path)
    auto_knn = joblib.load(auto_knn_path)
    baseline_knn = joblib.load(baseline_knn_path)
    scaler = joblib.load(scaler_path)
    embeddings = np.load(embeddings_path)
    df = pd.read_csv(df_path)
    print("Artifacts loaded successfully.")
    return df, encoder, auto_knn, baseline_knn, scaler, embeddings


# --- Main Block ---

if __name__ == "__main__":
    df, encoder, auto_knn, baseline_knn, scaler, embeddings = load_artifacts()

    seed_track = "Shape of You"
    mood = "happy"

    print("\n=== Evaluation Metrics ===")
    # Autoencoder metrics
    gc_auto = evaluate_genre_consistency(df, encoder, auto_knn, seed_track, use_encoder=True, scaler=scaler)
    ma_auto = evaluate_mood_alignment(df, encoder, auto_knn, mood, use_encoder=True, scaler=scaler)
    div_auto = evaluate_diversity(df, encoder, auto_knn, seed_track, use_encoder=True, scaler=scaler)

    # Baseline metrics
    gc_base = evaluate_genre_consistency(df, encoder, baseline_knn, seed_track, use_encoder=False, scaler=scaler)
    ma_base = evaluate_mood_alignment(df, encoder, baseline_knn, mood, use_encoder=False, scaler=scaler)
    div_base = evaluate_diversity(df, encoder, baseline_knn, seed_track, use_encoder=False, scaler=scaler)

    # Print comparison table
    print("\nMetric              Baseline KNN     Autoencoder KNN")
    print(f"Genre consistency   {gc_base:.2f}            {gc_auto:.2f}")
    print(f"Mood alignment      {ma_base:.2f}            {ma_auto:.2f}")
    print(f"Diversity           {div_base:.2f}            {div_auto:.2f}")
