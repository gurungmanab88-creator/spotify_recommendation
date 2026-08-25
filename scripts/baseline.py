import pandas as pd 
from sklearn.neighbors import NearestNeighbors 
import joblib
from config import AUDIO_FEATURES


def build_baseline(
    scaled_path = "outputs/scaled_features.csv",
    model_path = "outputs/baseline_knn.pkl",
    n_neighbors = 10
):
    df = pd.read_csv(scaled_path)
    X = df[AUDIO_FEATURES].values

    knn = NearestNeighbors(n_neighbors = n_neighbors, metric = "cosine")
    knn.fit(X)

    joblib.dump(knn , model_path)
    print(f"saved baseline model to {model_path}")

    return df, knn

def recommend(track_name, df ,knn , n_neighbors = 10):
    track = df[df["track_name"].str.lower() == track_name.lower()]
    if track.empty:
        print("Track not found.exe")
        return

    idx = track.index[0]
    X = df[AUDIO_FEATURES].values
    distances, indices = knn.kneighbors([X[idx]], n_neighbors = n_neighbors)


    recs = df.iloc[indices[0]]
    print(f"\n REcommendation for '{track_name}':")
    print(recs[["track_name", "artists", "track_genre", "popularity"]])

if __name__ == "__main__":
    df, knn = build_baseline()
    print("Baseline KNN model trained and saved. No user input required.")