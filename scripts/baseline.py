import pandas as pd 
from sklearn.neighbors import NearestNeighbors 
import joblib
from config import AUDIO_FEATURES

def build_baseline(
    train_path="outputs/train.csv",
    model_path="outputs/baseline_knn.pkl",
    n_neighbors=10
):
    # Load training data
    train_df = pd.read_csv(train_path)
    X_train = train_df[AUDIO_FEATURES].values

    # Fit KNN on train set
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    knn.fit(X_train)

    joblib.dump(knn, model_path)
    print(f"Baseline KNN model trained and saved to {model_path}")

    return train_df, knn

def recommend(track_name, df, knn, test_path="outputs/test.csv", n_neighbors=10):
    # Load test set for evaluation
    test_df = pd.read_csv(test_path)

    track = test_df[test_df["track_name"].str.lower() == track_name.lower()]
    if track.empty:
        print("Track not found in test set.")
        return

    idx = track.index[0]
    X_test = test_df[AUDIO_FEATURES].values
    distances, indices = knn.kneighbors([X_test[idx]], n_neighbors=n_neighbors)

    recs = test_df.iloc[indices[0]]
    print(f"\nBaseline recommendations for '{track_name}' (evaluated on test set):")
    print(recs[["track_name", "artists", "track_genre", "popularity"]])

if __name__ == "__main__":
    train_df, knn = build_baseline()
    print("Baseline KNN model trained on train set.")
    recommend("Shape of You", train_df, knn)
