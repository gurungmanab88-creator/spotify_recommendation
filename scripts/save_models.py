import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

def save_all(
    df,
    scaler,
    encoder,
    autoencoder,
    embeddings,
    auto_knn,
    baseline_knn,
    cluster_model,
    output_dir="outputs"
):
    
    df.to_csv(f"{output_dir}/cleaned_data.csv", index=False)
    print(f"Dataframe saved to {output_dir}/cleaned_data.csv")

    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    print(f"Scaler saved to {output_dir}/scaler.pkl")

    encoder.save(f"{output_dir}/encoder.keras")
    print(f"Encoder model saved to {output_dir}/encoder.keras")

    autoencoder.save(f"{output_dir}/autoencoder.keras")
    print(f"Autoencoder model saved to {output_dir}/autoencoder.keras")

    np.save(f"{output_dir}/embeddings.npy", embeddings)
    print(f"Embeddings saved to {output_dir}/embeddings.npy")

    joblib.dump(auto_knn, f"{output_dir}/autoencoder_knn.pkl")
    print(f"Autoencoder KNN saved to {output_dir}/autoencoder_knn.pkl")

    joblib.dump(baseline_knn, f"{output_dir}/baseline_knn.pkl")
    print(f"Baseline KNN saved to {output_dir}/baseline_knn.pkl")

    
    joblib.dump(cluster_model, f"{output_dir}/cluster_model.pkl")
    print(f"Best cluster model saved to {output_dir}/cluster_model.pkl")


if __name__ == "__main__":
    df = pd.read_csv("outputs/cleaned_data.csv")
    scaler = joblib.load("outputs/scaler.pkl")
    encoder = load_model("outputs/encoder.keras")
    autoencoder = load_model("outputs/autoencoder.keras")
    embeddings = np.load("outputs/embeddings.npy")
    auto_knn = joblib.load("outputs/autoencoder_knn.pkl")
    baseline_knn = joblib.load("outputs/baseline_knn.pkl")
    cluster_model = joblib.load("outputs/cluster_model.pkl")

    save_all(df, scaler, encoder, autoencoder, embeddings, auto_knn, baseline_knn, cluster_model)
