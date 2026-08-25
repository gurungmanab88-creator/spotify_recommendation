from baseline import AUDIO_FEATURES
from mood_parser import mood_to_vector
import pandas as pd
import numpy as np
from tensorflow.keras import layers, models
from sklearn.neighbors import NearestNeighbors
import joblib

AUDIO_FEATURES = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "liveness",
    "speechiness", "loudness"
]

def train_autoencoder(
    scaled_path="outputs/scaled_features.csv",
    encoder_path="outputs/encoder.keras",
    knn_path="outputs/autoencoder_knn.pkl",
    embedding_dim=10,
    n_neighbors=10,
    epochs=100,
    batch_size=32
):
    # Load scaled dataset
    df = pd.read_csv(scaled_path)
    X = df[AUDIO_FEATURES].values

    # Build autoencoder
    input_dim = X.shape[1]
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(16, activation="relu")(input_layer)
    encoded = layers.Dense(embedding_dim, activation="linear")(encoded)

    decoded = layers.Dense(16, activation="relu")(encoded)
    decoded = layers.Dense(input_dim, activation="linear")(decoded)

    autoencoder = models.Model(input_layer, decoded)
    encoder = models.Model(input_layer, encoded)

    autoencoder.compile(optimizer="adam", loss="mse")

    # Train autoencoder
    autoencoder.fit(
        X, X,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.1
    )

    # Save encoder
    encoder.save(encoder_path)
    print(f"Encoder saved to {encoder_path}")

    # Extract embeddings
    embeddings = encoder.predict(X)

    # Fit NearestNeighbors on embeddings
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    knn.fit(embeddings)
    joblib.dump(knn, knn_path)
    print(f"KNN model saved to {knn_path}")

    return df, encoder, knn

        # Save embeddings array
    np.save("outputs/embeddings.npy", embeddings)
    print("Embeddings saved to outputs/embeddings.npy")

    # Save cleaned dataframe for later use in Streamlit
    df.to_csv("outputs/cleaned_data.csv", index=False)
    print("Cleaned dataframe saved to outputs/cleaned_data.csv")



def recommend(track_name, df, encoder, knn, n_neighbors=10):
    track = df[df["track_name"].str.lower() == track_name.lower()]
    if track.empty:
        print("Track not found.")
        return

    idx = track.index[0]
    X = df[AUDIO_FEATURES].values
    embedding = encoder.predict(X[idx].reshape(1, -1))

    distances, indices = knn.kneighbors(embedding, n_neighbors=n_neighbors)
    recs = df.iloc[indices[0]]

    print(f"\nAutoencoder recommendations for '{track_name}':")
    print(recs[["track_name", "artists", "track_genre", "popularity"]])


def recommend_mood(mood, df, encoder, knn, n_neighbors=10, genre=None):
    vec = mood_to_vector(mood)
    if vec is None:
        return

    embedding = encoder.predict(vec)
    distances, indices = knn.kneighbors(embedding, n_neighbors=n_neighbors)
    recs = df.iloc[indices[0]]

    if genre:
        recs = recs[recs["track_genre"].str.lower() == genre.lower()]

    print(f"\nRecommendations for mood '{mood}'" + (f" in genre '{genre}'" if genre else "") + ":")
    print(recs[["track_name", "artists", "track_genre", "popularity"]])


if __name__ == "__main__":
    df, encoder, knn = train_autoencoder()
    while True:
        user_input = input("\nEnter a track name or mood (optionally add genre, or 'exit' to quit): ")
        if user_input.lower() == "exit":
            break

        parts = user_input.split()
        if len(parts) == 1:
            query = parts[0]
            genre = None
        else:
            query = parts[0]
            genre = " ".join(parts[1:])

        
        track = df[df["track_name"].str.lower() == query.lower()]
        if not track.empty:
            recommend(query, df, encoder, knn)
        else:
            recommend_mood(query, df, encoder, knn, genre=genre)
