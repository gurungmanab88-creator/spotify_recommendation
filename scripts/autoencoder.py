from baseline import AUDIO_FEATURES
import pandas as pd
import numpy as np
import tensorflow as keras
from tensorflow.keras import layers, models 
from sklearn.neighbors import NearestNeighbors
import joblib

AUDIO_FEATURES = [
    "danceability", "energy", "valence", "tempo", 
    "acousticness", "instrumentalness", "liveness", 
    "speechiness", "loudness"
]


def train_autoencoder(
    scaled_path = "outputs/scaled_features.csv",
    encoder_path = "outputs/encoder.h5",
    knn_path = "outputs/autoencoder_knn.pkl",
    embedding_dim = 5,
    n_neighbors = 10,
    epochs = 50,
    batch_size= 32
):
   # loading the scaled dataset
    df = pd.read_csv(scaled_path)
    X = df[AUDIO_FEATURES].values 

    #BUildiing the actual autoencoder

    input_dim = X.shape[1]
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(16, activation = "relu")(input_layer)
    encoded = layers.Dense(embedding_dim, activation="linear")(encoded)

    decoded = layers.Dense(16, activation = "relu")(encoded)
    decoded = layers.Dense(input_dim, activation="linear")(decoded)

    autoencoder = models.Model(input_layer, decoded)
    encoder = models.Model(input_layer, encoded)

    autoencoder.compile(optimizer = "adam", loss="mse")

    # Training the autoencoder

    autoencoder.fit(X, X, 
    epochs = epochs , 
    batch_size = batch_size ,
    shuffle = True,
    validation_split = 0.1
    )

    # saving the trained encoder

    encoder.save (encoder_path)
    print(f"encoder saved to {encoder_path}")

    # Extractin the embeddings
    embeddings = encoder.predict(X)


    #Fitting NeareastNeighbors on embedding

    knn = NearestNeighbors(n_neighbors = n_neighbors , metric = "cosine")
    knn.fit(embeddings)
    joblib.dump(knn , knn_path)
    print(f"the autoencoder knn model saved to {knn_path}")

    return df, encoder , knn 


def recommend(track_name , df, encoder,knn, n_neighbors = 10 ):
    track = df[df["track_name"].str.lower() == track_name.lower()]
    if track.empty:
        print("Track not found exe")
        return
    
    idx = track.index[0]
    X = df[AUDIO_FEATURES].values
    embedding = encoder.predict(X[idx].reshape(1, -1)) 


    distances , indices = knn.kneighbors(embedding, n_neighbors = n_neighbors)
    recs = df.iloc[indices[0]]

    print(f"\n Autoencoder recommendations for '{track_name}':")
    print(recs[["track_name", "artists", "track_genre", "popularity"]])


if __name__ == "__main__":
    df , encoder , knn = train_autoencoder()
    track_name = input("Enter the track name : ")

    recommend(track_name, df, encoder,knn)



   






