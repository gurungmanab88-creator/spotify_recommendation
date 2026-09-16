import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def preprocess(
    input_path="data/spotify_tracks.csv",
    cleaned_path="outputs/cleaned_tracks.csv",
    scaled_path="outputs/scaled_features.csv",
    scaler_path="outputs/scaler.pkl",
    train_path="outputs/train.csv",
    val_path="outputs/val.csv",
    test_path="outputs/test.csv"
):
    os.makedirs("outputs", exist_ok=True)

    # Load dataset
    df = pd.read_csv(input_path)

    # Drop stray index column
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    audio_features = [
        "danceability", "energy", "valence", "tempo",
        "acousticness", "instrumentalness", "liveness",
        "speechiness", "loudness"
    ]

    # Drop rows with missing values in key columns
    df = df.dropna(subset=["track_name", "artists"] + audio_features)

    # Keep highest popularity per track/artist
    df = df.sort_values("popularity", ascending=False)
    df = df.drop_duplicates(subset=["track_name", "artists"], keep="first")

    # Save cleaned dataset
    df.to_csv(cleaned_path, index=False)
    print(f"Cleaned dataset saved to {cleaned_path}")

    # Scale audio features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[audio_features])

    # Create scaled DataFrame
    scaled_df = pd.DataFrame(scaled_features, columns=audio_features, index=df.index)

    # Add metadata back
    scaled_df["track_name"] = df["track_name"]
    scaled_df["artists"] = df["artists"]
    scaled_df["popularity"] = df["popularity"]
    scaled_df["track_genre"] = df["track_genre"]

    # Save scaled dataset
    scaled_df.to_csv(scaled_path, index=False)
    print(f"Scaled dataset saved to {scaled_path}")

    # Save scaler object
    joblib.dump(scaler, scaler_path)
    print(f"Scaler object saved to {scaler_path}")

    # Train/Val/Test split (70/15/15)
    train_df, temp_df = train_test_split(scaled_df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train set saved to {train_path}")
    print(f"Validation set saved to {val_path}")
    print(f"Test set saved to {test_path}")

if __name__ == "__main__":
    preprocess()
