from config import AUDIO_FEATURES
from mood_parser import mood_to_vector

import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.neighbors import NearestNeighbors

import joblib
import os
import pickle


np.random.seed(42)
tf.random.set_seed(42)



def train_autoencoder(
    train_path="outputs/train.csv",
    val_path="outputs/val.csv",
    test_path="outputs/test.csv",
    encoder_path="outputs/encoder.keras",
    autoencoder_path="outputs/autoencoder.keras",
    knn_path="outputs/autoencoder_knn.pkl",
    embedding_dim=8,
    n_neighbors=10,
    epochs=50,
    batch_size=64
):

    

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[AUDIO_FEATURES].values.astype(np.float32)
    X_val = val_df[AUDIO_FEATURES].values.astype(np.float32)
    X_test = test_df[AUDIO_FEATURES].values.astype(np.float32)

    print("\n========================================")
    print("AUTOENCODER TRAINING")
    print("========================================")
    print(f"Training samples   : {len(X_train)}")
    print(f"Validation samples : {len(X_val)}")
    print(f"Test samples       : {len(X_test)}")
    print(f"Input features     : {X_train.shape[1]}")
    print(f"Embedding size     : {embedding_dim}")
    print(f"Maximum epochs     : {epochs}")
    print(f"Batch size         : {batch_size}")
    print("========================================\n")


   

    input_dim = X_train.shape[1]

    input_layer = layers.Input(
        shape=(input_dim,),
        name="audio_input"
    )

    # Encoder
    encoded_hidden = layers.Dense(
        16,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="encoder_hidden"
    )(input_layer)

    encoded_hidden = layers.Dropout(
        0.2,
        name="encoder_dropout"
    )(encoded_hidden)

    # Actual embedding layer
    embedding_layer = layers.Dense(
        embedding_dim,
        activation="linear",
        name="embedding"
    )(encoded_hidden)

    # Decoder
    decoded_hidden = layers.Dense(
        16,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="decoder_hidden"
    )(embedding_layer)

    decoded_hidden = layers.Dropout(
        0.2,
        name="decoder_dropout"
    )(decoded_hidden)

    output_layer = layers.Dense(
        input_dim,
        activation="linear",
        name="reconstruction"
    )(decoded_hidden)

    # Full autoencoder
    autoencoder = models.Model(
        inputs=input_layer,
        outputs=output_layer,
        name="music_autoencoder"
    )

    # Encoder model
    encoder = models.Model(
        inputs=input_layer,
        outputs=embedding_layer,
        name="music_encoder"
    )

    # Compile
    autoencoder.compile(
        optimizer="adam",
        loss="mse"
    )

    autoencoder.summary()


    

    checkpoint_path = "outputs/autoencoder_checkpoint.keras"

    checkpoint_cb = ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    )



    print("\nStarting autoencoder training...\n")

    history = autoencoder.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(X_val, X_val),
        callbacks=[checkpoint_cb],
        verbose=1
    )


    

    history_path = "outputs/autoencoder_history.pkl"

    with open(history_path, "wb") as f:
        pickle.dump(history.history, f)

    print(
        f"\nTraining history saved to "
        f"{history_path}"
    )


    
    if os.path.exists(checkpoint_path):

        print(
            "\nLoading best autoencoder checkpoint "
            "based on validation loss..."
        )

        best_autoencoder = load_model(
            checkpoint_path
        )

        # Recreate encoder using the named embedding layer
        encoder = models.Model(
            inputs=best_autoencoder.input,
            outputs=best_autoencoder.get_layer(
                "embedding"
            ).output,
            name="music_encoder"
        )

        autoencoder = best_autoencoder

    else:

        print(
            "\nCheckpoint not found. "
            "Using current trained model."
        )


   

    encoder.save(encoder_path)
    autoencoder.save(autoencoder_path)

    print(
        f"Encoder saved to {encoder_path}"
    )

    print(
        f"Autoencoder saved to {autoencoder_path}"
    )


  

    print("\nCreating train embeddings...")

    train_embeddings = encoder.predict(
        X_train,
        verbose=1
    )

    np.save(
        "outputs/train_embeddings.npy",
        train_embeddings
    )



    print("Creating validation embeddings...")

    val_embeddings = encoder.predict(
        X_val,
        verbose=1
    )

    np.save(
        "outputs/val_embeddings.npy",
        val_embeddings
    )


   

    print("Creating test embeddings...")

    test_embeddings = encoder.predict(
        X_test,
        verbose=1
    )

    np.save(
        "outputs/test_embeddings.npy",
        test_embeddings
    )



    full_df = pd.concat(
        [
            train_df,
            val_df,
            test_df
        ],
        ignore_index=True
    )

    X_full = full_df[AUDIO_FEATURES].values.astype(
        np.float32
    )


  

    print("\nCreating full dataset embeddings...")

    embeddings = encoder.predict(
        X_full,
        verbose=1
    )


    

    print("\nTraining KNN on full embeddings...")

    knn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine"
    )

    knn.fit(embeddings)

    joblib.dump(
        knn,
        knn_path
    )

    print(
        f"KNN model saved to {knn_path}"
    )




    np.save(
        "outputs/embeddings.npy",
        embeddings
    )

    print(
        "Embeddings saved to outputs/embeddings.npy"
    )


    

    full_df.to_csv(
        "outputs/cleaned_data.csv",
        index=False
    )

    print(
        "Cleaned dataframe saved to "
        "outputs/cleaned_data.csv"
    )


    

    print("\n========================================")
    print("AUTOENCODER TRAINING COMPLETE")
    print("========================================")
    print(f"Train embeddings : {train_embeddings.shape}")
    print(f"Val embeddings   : {val_embeddings.shape}")
    print(f"Test embeddings  : {test_embeddings.shape}")
    print(f"Full embeddings  : {embeddings.shape}")
    print("========================================\n")


    return full_df, encoder, knn


    

def recommend(
    track_name,
    df,
    encoder,
    knn,
    n_neighbors=10
):

    
    df = df.reset_index(drop=True)

    track = df[
        df["track_name"]
        .astype(str)
        .str.lower()
        == track_name.lower()
    ]

    if track.empty:
        print("Track not found.")
        return

    
    idx = track.index[0]

    X = df[AUDIO_FEATURES].values.astype(
        np.float32
    )

   
    embedding = encoder.predict(
        X[idx].reshape(1, -1),
        verbose=0
    )

    
    distances, indices = knn.kneighbors(
        embedding,
        n_neighbors=n_neighbors
    )

    recs = df.iloc[indices[0]].copy()

    
    recs["distance"] = distances[0]
    recs["match_score"] = (
        1 - recs["distance"]
    ) * 100

    print(
        f"\nAutoencoder recommendations "
        f"for '{track_name}':"
    )

    columns = [
        "track_name",
        "artists",
        "track_genre",
        "popularity",
        "match_score"
    ]

    available_columns = [
        col for col in columns
        if col in recs.columns
    ]

    print(
        recs[available_columns]
        .to_string(index=False)
    )




def recommend_mood(
    mood,
    df,
    encoder,
    knn,
    n_neighbors=10,
    genre=None
):

    vec = mood_to_vector(mood)

    if vec is None:
        print(
            f"Could not convert mood '{mood}' "
            "to a vector."
        )
        return

    vec = np.asarray(
        vec,
        dtype=np.float32
    )

    
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)

    embedding = encoder.predict(
        vec,
        verbose=0
    )

    distances, indices = knn.kneighbors(
        embedding,
        n_neighbors=n_neighbors
    )

    recs = df.iloc[indices[0]].copy()

    recs["distance"] = distances[0]

    recs["match_score"] = (
        1 - recs["distance"]
    ) * 100

    
    if genre:

        recs = recs[
            recs["track_genre"]
            .astype(str)
            .str.lower()
            == genre.lower()
        ]

    print(
        f"\nRecommendations for mood "
        f"'{mood}'"
        + (
            f" in genre '{genre}'"
            if genre
            else ""
        )
        + ":"
    )

    columns = [
        "track_name",
        "artists",
        "track_genre",
        "popularity",
        "match_score"
    ]

    available_columns = [
        col for col in columns
        if col in recs.columns
    ]

    print(
        recs[available_columns]
        .to_string(index=False)
    )




if __name__ == "__main__":

    df, encoder, knn = train_autoencoder(
        epochs=50,
        batch_size=64
    )

    print(
        "Autoencoder trained, "
        "encoder/embeddings/KNN saved to outputs/. "
        "No user input required."
    )