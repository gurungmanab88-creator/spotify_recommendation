import numpy as np
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def cluster_embeddings(
    train_embeddings_path="outputs/train_embeddings.npy",
    val_embeddings_path="outputs/val_embeddings.npy",
    test_embeddings_path="outputs/test_embeddings.npy",
    train_df_path="outputs/train.csv",
    val_df_path="outputs/val.csv",
    test_df_path="outputs/test.csv",
    kmeans_path="outputs/kmeans.pkl",
    gmm_path="outputs/gmm.pkl",
    final_path="outputs/cluster_model.pkl",
    n_clusters=20
):
    
    train_embeddings = np.load(train_embeddings_path)
    val_embeddings = np.load(val_embeddings_path)
    test_embeddings = np.load(test_embeddings_path)

    print(f"Train embeddings shape: {train_embeddings.shape}")
    print(f"Val embeddings shape: {val_embeddings.shape}")
    print(f"Test embeddings shape: {test_embeddings.shape}")

    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(train_embeddings)
    score_kmeans = silhouette_score(val_embeddings, kmeans.predict(val_embeddings))
    joblib.dump(kmeans, kmeans_path)
    print(f"KMeans saved to {kmeans_path} (val silhouette={score_kmeans:.3f})")

    
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(train_embeddings)
    score_gmm = silhouette_score(val_embeddings, gmm.predict(val_embeddings))
    joblib.dump(gmm, gmm_path)
    print(f"GMM saved to {gmm_path} (val silhouette={score_gmm:.3f})")

  
    if score_kmeans >= score_gmm:
        print("KMeans performed better — using cluster_model.pkl")
        best_model = kmeans
    else:
        print("GMM performed better — using cluster_model.pkl")
        best_model = gmm
    joblib.dump(best_model, final_path)

    
    train_df = pd.read_csv(train_df_path)
    val_df = pd.read_csv(val_df_path)
    test_df = pd.read_csv(test_df_path)

    train_df["cluster"] = best_model.predict(train_embeddings)
    val_df["cluster"] = best_model.predict(val_embeddings)
    test_df["cluster"] = best_model.predict(test_embeddings)

    train_df.to_csv("outputs/train_with_clusters.csv", index=False)
    val_df.to_csv("outputs/val_with_clusters.csv", index=False)
    test_df.to_csv("outputs/test_with_clusters.csv", index=False)

    print("Cluster labels saved for train, val, and test splits.")

if __name__ == "__main__":
    cluster_embeddings()
