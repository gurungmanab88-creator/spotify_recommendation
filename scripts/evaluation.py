import pandas as pd
import numpy as np
import joblib
import random
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, silhouette_score, davies_bouldin_score
from config import AUDIO_FEATURES
from mood_parser import mood_to_vector



def evaluate_genre_consistency(df, encoder, knn, track_name, n_neighbors=10, use_encoder=False, scaler=None):
    track = df[df["track_name"].str.lower() == track_name.lower()]
    if track.empty:
        return None
    genre = track["track_genre"].values[0]
    idx = track.index[0]
    X = df[AUDIO_FEATURES].values
    query = encoder.predict(X[idx].reshape(1, -1)) if use_encoder else scaler.transform(X[idx].reshape(1, -1))
    distances, indices = knn.kneighbors(query, n_neighbors=n_neighbors)
    recs = df.iloc[indices[0]]
    same_genre = (recs["track_genre"].str.lower() == genre.lower()).sum()
    return same_genre / n_neighbors


def evaluate_mood_alignment(df, encoder, knn, mood, n_neighbors=10, use_encoder=False, scaler=None):
    vec = mood_to_vector(mood)
    if vec is None:
        return None
    query = encoder.predict(vec) if use_encoder else vec
    distances, indices = knn.kneighbors(query, n_neighbors=n_neighbors)
    recs = df.iloc[indices[0]]
    avg_features = recs[AUDIO_FEATURES].mean().values
    similarity = np.dot(vec.flatten(), avg_features) / (
        np.linalg.norm(vec.flatten()) * np.linalg.norm(avg_features)
    )
    return similarity


def evaluate_diversity(df, encoder, knn, track_name, n_neighbors=10, use_encoder=False, scaler=None):
    track = df[df["track_name"].str.lower() == track_name.lower()]
    if track.empty:
        return None
    idx = track.index[0]
    X = df[AUDIO_FEATURES].values
    query = encoder.predict(X[idx].reshape(1, -1)) if use_encoder else scaler.transform(X[idx].reshape(1, -1))
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
    return diversity


def evaluate_reconstruction_error(autoencoder, test_df):
    X_test = test_df[AUDIO_FEATURES].values
    X_recon = autoencoder.predict(X_test)
    mse = mean_squared_error(X_test.flatten(), X_recon.flatten())
    print(f"Reconstruction MSE on test set: {mse:.4f}")
    return mse


def evaluate_clustering_quality(embeddings, cluster_model, split_name="val"):
    labels = cluster_model.predict(embeddings)
    sil = silhouette_score(embeddings, labels)
    db = davies_bouldin_score(embeddings, labels)
    print(f"{split_name} Silhouette: {sil:.3f}, Davies–Bouldin: {db:.3f}")
    return sil, db


def evaluate_retrieval_quality_auto(df, encoder, knn, k=10):
    tracks = df["track_name"].dropna().unique().tolist()
    precisions, recalls = [], []
    for track_name in random.sample(tracks, min(30, len(tracks))):
        track = df[df["track_name"].str.lower() == track_name.lower()]
        if track.empty: 
            continue
        genre = track["track_genre"].values[0]
        idx = track.index[0]
        X = df[AUDIO_FEATURES].values
        
        embedding = encoder.predict(X[idx].reshape(1, -1))
        distances, indices = knn.kneighbors(embedding, n_neighbors=k)
        recs = df.iloc[indices[0]]
        same_genre = (recs["track_genre"].str.lower() == genre.lower()).sum()
        precisions.append(same_genre / k)
        recalls.append(same_genre / len(df[df["track_genre"].str.lower() == genre.lower()]))
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    print(f"[Autoencoder KNN] Precision@{k}: {avg_precision:.3f}, Recall@{k}: {avg_recall:.3f}")
    return avg_precision, avg_recall


def evaluate_retrieval_quality_baseline(df, scaler, knn, k=10):
    tracks = df["track_name"].dropna().unique().tolist()
    precisions, recalls = [], []
    for track_name in random.sample(tracks, min(30, len(tracks))):
        track = df[df["track_name"].str.lower() == track_name.lower()]
        if track.empty: 
            continue
        genre = track["track_genre"].values[0]
        idx = track.index[0]
        X = df[AUDIO_FEATURES].values
        
        query = scaler.transform(pd.DataFrame([X[idx]], columns=AUDIO_FEATURES))
        distances, indices = knn.kneighbors(query, n_neighbors=k)
        recs = df.iloc[indices[0]]
        same_genre = (recs["track_genre"].str.lower() == genre.lower()).sum()
        precisions.append(same_genre / k)
        recalls.append(same_genre / len(df[df["track_genre"].str.lower() == genre.lower()]))
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    print(f"[Baseline KNN] Precision@{k}: {avg_precision:.3f}, Recall@{k}: {avg_recall:.3f}")
    return avg_precision, avg_recall




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


if __name__ == "__main__":
    df, encoder, auto_knn, baseline_knn, scaler, embeddings = load_artifacts()
    autoencoder = load_model("outputs/autoencoder.keras")

   
    evaluate_reconstruction_error(autoencoder, df)

  
    evaluate_clustering_quality(embeddings, joblib.load("outputs/cluster_model.pkl"), split_name="full")

    
    random_track = random.choice(df["track_name"].dropna().unique().tolist())
    print(f"Using random track: {random_track}")

    print("Genre consistency:", evaluate_genre_consistency(df, encoder, auto_knn, random_track, use_encoder=True))
    print("Mood alignment:", evaluate_mood_alignment(df, encoder, auto_knn, "happy", use_encoder=True))
    print("Diversity:", evaluate_diversity(df, encoder, auto_knn, random_track, use_encoder=True))

    
    auto_prec, auto_rec = evaluate_retrieval_quality_auto(df, encoder, auto_knn, k=10)
    base_prec, base_rec = evaluate_retrieval_quality_baseline(df, scaler, baseline_knn, k=10)

    print("\n--- Retrieval Comparison ---")
    print(f"Autoencoder KNN: Precision={auto_prec:.3f}, Recall={auto_rec:.3f}")
    print(f"Baseline KNN:    Precision={base_prec:.3f}, Recall={base_rec:.3f}")


