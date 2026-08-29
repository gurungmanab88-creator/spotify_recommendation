import numpy as np
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def cluster_embeddings(
    embeddings_path="outputs/embeddings.npy",
    df_path="outputs/cleaned_data.csv",
    kmeans_path="outputs/kmeans.pkl",
    gmm_path="outputs/gmm.pkl",
    final_path="outputs/cluster_model.pkl",
    n_clusters=10
):
    embeddings = np.load(embeddings_path)
    df = pd.read_csv(df_path)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels_kmeans = kmeans.fit_predict(embeddings)
    score_kmeans = silhouette_score(embeddings, labels_kmeans)
    joblib.dump(kmeans, kmeans_path)
    print(f"KMeans saved to {kmeans_path} (silhouette={score_kmeans:.3f})")

    
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    labels_gmm = gmm.fit_predict(embeddings)
    score_gmm = silhouette_score(embeddings, labels_gmm)
    joblib.dump(gmm, gmm_path)
    print(f"GMM saved to {gmm_path} (silhouette={score_gmm:.3f})")

    if score_kmeans >= score_gmm:
        print("KMeans performed better — using cluster_model.pkl")
        df["cluster"] = labels_kmeans
        joblib.dump(kmeans, final_path) 
    else:
        print("GMM performed better — using cluster_model.pkl")
        df["cluster"] = labels_gmm
        joblib.dump(gmm, final_path)     

    df.to_csv(df_path, index=False)
    return df

if __name__ == "__main__":
    cluster_embeddings()
