import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def cluster_embeddings(
    embeddings_path="outputs/embeddings.npy",
    kmeans_path="outputs/kmeans.pkl",
    gmm_path="outputs/gmm.pkl",
    n_clusters=10
):
    # Load embeddings
    embeddings = np.load(embeddings_path)

    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels_kmeans = kmeans.fit_predict(embeddings)
    score_kmeans = silhouette_score(embeddings, labels_kmeans)
    joblib.dump(kmeans, kmeans_path)
    print(f"KMeans saved to {kmeans_path} (silhouette={score_kmeans:.3f})")

    # Fit GMM
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    labels_gmm = gmm.fit_predict(embeddings)
    score_gmm = silhouette_score(embeddings, labels_gmm)
    joblib.dump(gmm, gmm_path)
    print(f"GMM saved to {gmm_path} (silhouette={score_gmm:.3f})")

    # Pick best model
    if score_kmeans >= score_gmm:
        print("KMeans performed better — use kmeans.pkl")
        return kmeans, labels_kmeans
    else:
        print("GMM performed better — use gmm.pkl")
        return gmm, labels_gmm

if __name__ == "__main__":
    cluster_embeddings()
