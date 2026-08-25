import numpy as np
from mood_parser import mood_to_vector
from baseline import AUDIO_FEATURES

def evaluate_genre_consistency(df, encoder, knn, track_name, n_neighbors=10):
    track = df[df["track_name"].str.lower() == track_name.lower()]
    if track.empty:
        print("Track not found.")
        return None

    genre = track["track_genre"].values[0]
    idx = track.index[0]
    X = df[AUDIO_FEATURES].values
    embedding = encoder.predict(X[idx].reshape(1, -1))

    distances, indices = knn.kneighbors(embedding, n_neighbors=n_neighbors)
    recs = df.iloc[indices[0]]

    same_genre = (recs["track_genre"].str.lower() == genre.lower()).sum()
    consistency = same_genre / n_neighbors
    print(f"Genre consistency for '{track_name}' ({genre}): {consistency:.2f}")
    return consistency


def evaluate_mood_alignment(df, encoder, knn, mood, n_neighbors=10):
    vec = mood_to_vector(mood)
    if vec is None:
        return None

    embedding = encoder.predict(vec)
    distances, indices = knn.kneighbors(embedding, n_neighbors=n_neighbors)
    recs = df.iloc[indices[0]]

    avg_features = recs[AUDIO_FEATURES].mean().values
    similarity = np.dot(vec.flatten(), avg_features) / (
        np.linalg.norm(vec.flatten()) * np.linalg.norm(avg_features)
    )
    print(f"Mood alignment for '{mood}': {similarity:.2f}")
    return similarity

def evaluate_diversity(df, encoder, knn, track_name, n_neighbors=10):
    track = df[df["track_name"].str.lower() == track_name.lower()]
    if track.empty:
        print("Track not found.")
        return None

    idx = track.index[0]
    X = df[AUDIO_FEATURES].values
    embedding = encoder.predict(X[idx].reshape(1, -1))

    distances, indices = knn.kneighbors(embedding, n_neighbors=n_neighbors)
    recs = df.iloc[indices[0]]

    rec_embeddings = encoder.predict(recs[AUDIO_FEATURES].values)
    sim_matrix = np.dot(rec_embeddings, rec_embeddings.T)
    norms = np.linalg.norm(rec_embeddings, axis=1)
    cosine_sim = sim_matrix / np.outer(norms, norms)


    mask = ~np.eye(cosine_sim.shape[0], dtype=bool)
    diversity = 1 - cosine_sim[mask].mean()
    print(f"Diversity for '{track_name}': {diversity:.2f}")
    return diversity
