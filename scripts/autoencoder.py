from config import AUDIO_FEATURES
from mood_parser import mood_to_vector
import pandas as pd
import numpy as np, tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.neighbors import NearestNeighbors
import joblib, os
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint


np.random.seed(42)
tf.random.set_seed(42)

def train_autoencoder(
    scaled_path="outputs/scaled_features.csv",
    encoder_path="outputs/encoder.keras",
    knn_path="outputs/autoencoder_knn.pkl",
    embedding_dim=10,
    n_neighbors=10,
    epochs=30,
    batch_size=32
):


 
    df = pd.read_csv(scaled_path)
    X = df[AUDIO_FEATURES].values

 
    if os.path.exists("outputs/encoder_checkpoint.keras"):
        autoencoder = load_model("outputs/encoder_checkpoint.keras")
        print("Resuming training from checkpoint...")
 
        encoder = models.Model(autoencoder.input, autoencoder.layers[-3].output)
    else:

        input_dim = X.shape[1]
        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(16, activation="relu")(input_layer)
        encoded = layers.Dense(embedding_dim, activation="linear")(encoded)
        decoded = layers.Dense(16, activation="relu")(encoded)
        decoded = layers.Dense(input_dim, activation="linear")(decoded)
        autoencoder = models.Model(input_layer, decoded)
        encoder = models.Model(input_layer, encoded)
        autoencoder.compile(optimizer="adam", loss="mse")


    checkpoint_cb = ModelCheckpoint(
        "outputs/encoder_checkpoint.keras",
        save_best_only=True,
        monitor="val_loss",
        mode="min"
    )

    history = autoencoder.fit(
        X, X,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.1,
        callbacks=[checkpoint_cb]  
    )

    import pickle
    with open("outputs/autoencoder_history.pkl", "wb") as f:
        pickle.dump(history.history, f)
    print("Training history saved to outputs/autoencoder_history.pkl")

    
    encoder.save(encoder_path)
    print(f"Encoder saved to {encoder_path}")

    embeddings = encoder.predict(X)

  
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    knn.fit(embeddings)
    joblib.dump(knn, knn_path)
    print(f"KNN model saved to {knn_path}")


    np.save("outputs/embeddings.npy", embeddings)
    print("Embeddings saved to outputs/embeddings.npy")

    
    df.to_csv("outputs/cleaned_data.csv", index=False)
    print("Cleaned dataframe saved to outputs/cleaned_data.csv")

    return df, encoder, knn



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
    print("Autoencoder trained, encoder/embeddings/KNN saved to outputs/. No user input required.")
