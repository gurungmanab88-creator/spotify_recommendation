import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scripts.config import AUDIO_FEATURES, OUTPUT_DIR

from scripts.eda import (
    plot_feature_distributions,
    plot_genre_counts,
    plot_correlation_heatmap,
    plot_popularity_distribution
)

# ----------------------- Mood lexicon -----------------------
MOOD_MAP = {
    "sad": {
        "danceability": 0.3, "energy": 0.2, "valence": 0.2, "tempo": 0.4,
        "acousticness": 0.7, "instrumentalness": 0.5, "liveness": 0.3,
        "speechiness": 0.2, "loudness": 0.3
    },
    "happy": {
        "danceability": 0.7, "energy": 0.8, "valence": 0.9, "tempo": 0.7,
        "acousticness": 0.3, "instrumentalness": 0.2, "liveness": 0.6,
        "speechiness": 0.4, "loudness": 0.8
    },
    "party": {
        "danceability": 0.9, "energy": 0.9, "valence": 0.85, "tempo": 0.8,
        "acousticness": 0.2, "instrumentalness": 0.1, "liveness": 0.7,
        "speechiness": 0.5, "loudness": 0.9
    },
    "chill": {
        "danceability": 0.4, "energy": 0.35, "valence": 0.5, "tempo": 0.4,
        "acousticness": 0.7, "instrumentalness": 0.7, "liveness": 0.3,
        "speechiness": 0.2, "loudness": 0.4
    },
    "angry": {
        "danceability": 0.5, "energy": 0.9, "valence": 0.2, "tempo": 0.7,
        "acousticness": 0.2, "instrumentalness": 0.3, "liveness": 0.5,
        "speechiness": 0.6, "loudness": 0.9
    },
    "romantic": {
        "danceability": 0.5, "energy": 0.4, "valence": 0.7, "tempo": 0.5,
        "acousticness": 0.6, "instrumentalness": 0.4, "liveness": 0.4,
        "speechiness": 0.3, "loudness": 0.5
    },
    "focus": {
        "danceability": 0.4, "energy": 0.4, "valence": 0.5, "tempo": 0.45,
        "acousticness": 0.5, "instrumentalness": 0.7, "liveness": 0.3,
        "speechiness": 0.2, "loudness": 0.4
    },
    "sleepy": {
        "danceability": 0.3, "energy": 0.15, "valence": 0.3, "tempo": 0.3,
        "acousticness": 0.8, "instrumentalness": 0.6, "liveness": 0.2,
        "speechiness": 0.2, "loudness": 0.2
    },
    "motivated": {
        "danceability": 0.7, "energy": 0.85, "valence": 0.75, "tempo": 0.7,
        "acousticness": 0.3, "instrumentalness": 0.3, "liveness": 0.5,
        "speechiness": 0.4, "loudness": 0.8
    },
    "calm": {
        "danceability": 0.4, "energy": 0.3, "valence": 0.55, "tempo": 0.4,
        "acousticness": 0.75, "instrumentalness": 0.6, "liveness": 0.3,
        "speechiness": 0.2, "loudness": 0.3
    },
    "epic": {
        "danceability": 0.6, "energy": 0.8, "valence": 0.6, "tempo": 0.8,
        "acousticness": 0.3, "instrumentalness": 0.4, "liveness": 0.6,
        "speechiness": 0.3, "loudness": 0.8
    },
    "melancholy": {
        "danceability": 0.3, "energy": 0.3, "valence": 0.25, "tempo": 0.4,
        "acousticness": 0.7, "instrumentalness": 0.5, "liveness": 0.3,
        "speechiness": 0.2, "loudness": 0.3
    },
    "uplifting": {
        "danceability": 0.7, "energy": 0.75, "valence": 0.85, "tempo": 0.7,
        "acousticness": 0.3, "instrumentalness": 0.3, "liveness": 0.6,
        "speechiness": 0.4, "loudness": 0.8
    },
    "dark": {
        "danceability": 0.4, "energy": 0.5, "valence": 0.2, "tempo": 0.5,
        "acousticness": 0.5, "instrumentalness": 0.6, "liveness": 0.4,
        "speechiness": 0.3, "loudness": 0.4
    },
    "energetic": {
        "danceability": 0.8, "energy": 0.9, "valence": 0.7, "tempo": 0.8,
        "acousticness": 0.2, "instrumentalness": 0.2, "liveness": 0.6,
        "speechiness": 0.4, "loudness": 0.9
    }
}


@st.cache_resource
def load_models():
    encoder = load_model(os.path.join(OUTPUT_DIR, "encoder.keras"))
    auto_knn = joblib.load(os.path.join(OUTPUT_DIR, "autoencoder_knn.pkl"))
    baseline_knn = joblib.load(os.path.join(OUTPUT_DIR, "baseline_knn.pkl"))
    cluster_model = joblib.load(os.path.join(OUTPUT_DIR, "cluster_model.pkl"))   # ✅ load winner
    return encoder, auto_knn, baseline_knn, cluster_model


@st.cache_data
def load_data():
    scaler = joblib.load(os.path.join(OUTPUT_DIR, "scaler.pkl"))
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "cleaned_data.csv"))
    embeddings = np.load(os.path.join(OUTPUT_DIR, "embeddings.npy"))
    return df, scaler, embeddings

encoder, auto_knn, baseline_knn, cluster_model = load_models()
df, scaler, embeddings = load_data()

# ----------------------- Plotting functions -----------------------
def smooth_curve(points, factor=0.9):
    smoothed = []
    for p in points:
        if smoothed:
            smoothed.append(smoothed[-1] * factor + p * (1 - factor))
        else:
            smoothed.append(p)
    return smoothed

def plot_loss_curves(history):
    train_loss = history["loss"]
    val_loss = history.get("val_loss", [])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_loss, label="Train Loss", alpha=0.5)
    if val_loss:
        ax.plot(val_loss, label="Val Loss", alpha=0.5)
        ax.plot(smooth_curve(val_loss), label="Val Loss (smoothed)")
    ax.plot(smooth_curve(train_loss), label="Train Loss (smoothed)")
    ax.set_title("Autoencoder Training Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    return fig

def plot_clusters(embeddings, labels, method="pca"):
    if method == "tsne":
        X_embedded = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
        title = "t-SNE Visualization of Song Clusters"
    else:
        X_embedded = PCA(n_components=2).fit_transform(embeddings)
        title = "PCA Visualization of Song Clusters"
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap="tab10", alpha=0.7)
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax, label="Cluster")
    return fig

def plot_mood_radar(mood_name, mood_vector, features=AUDIO_FEATURES):
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    values = list(mood_vector.values())
    values += values[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, "o-", linewidth=2, label=mood_name)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_title(f"Mood Profile: {mood_name}")
    ax.legend(loc="upper right")
    return fig

@st.cache_data
def cached_feature_distributions(_df):
    return plot_feature_distributions(_df)

@st.cache_data
def cached_genre_counts(_df):
    return plot_genre_counts(_df)

@st.cache_data
def cached_correlation_heatmap(_df):
    return plot_correlation_heatmap(_df)

@st.cache_data
def cached_popularity_distribution(_df):
    return plot_popularity_distribution(_df)

@st.cache_data
def cached_loss_curve(history):
    return plot_loss_curves(history)

@st.cache_data
def cached_cluster_plot(_embeddings, _labels, method="pca"):
    return plot_clusters(_embeddings, _labels, method=method)

@st.cache_data
def cached_mood_radar(mood_name, mood_vector):
    return plot_mood_radar(mood_name, mood_vector)

# ----------------------- Recommendation helpers -----------------------
def mood_to_vector(mood: str, scaler):
    mapping = MOOD_MAP.get(mood.lower())
    if mapping is None:
        return None
    
    vec = np.array([mapping.get(feat, 0.5) for feat in AUDIO_FEATURES]).reshape(1, -1)
    
    return scaler.transform(vec)

@st.cache_data
def get_recommendations(track_name, _df, _knn, n_neighbors=10, is_autoencoder=False, _encoder=None, _scaler=None):
    track = _df[_df["track_name"].str.lower() == track_name.lower()]
    if track.empty:
        return None

    idx = track.index[0]
    X = _df[AUDIO_FEATURES].values

    if is_autoencoder:
        embedding = _encoder.predict(X[idx].reshape(1, -1), verbose=0)
        distances, indices = _knn.kneighbors(embedding, n_neighbors=n_neighbors)
    else:
        query_df = pd.DataFrame(X[idx].reshape(1, -1), columns=AUDIO_FEATURES)
        query = _scaler.transform(query_df)
        distances, indices = _knn.kneighbors(query, n_neighbors=n_neighbors)

    recs = _df.iloc[indices[0]].copy()
    max_dist = distances[0].max()
    recs["match_score"] = 1 - (distances[0] / max_dist)
    return recs[["track_name", "artists", "track_genre", "popularity", "match_score"]]


@st.cache_data
def get_mood_recommendations(mood, _df, _encoder, _knn, _scaler, n_neighbors=10, genre=None):
    vec = mood_to_vector(mood, _scaler)
    if vec is None:
        return None

    embedding = _encoder.predict(vec, verbose=0)
    distances, indices = _knn.kneighbors(embedding, n_neighbors=n_neighbors * 3)
    recs = _df.iloc[indices[0]].copy()
    max_dist = distances[0].max()
    recs["match_score"] = 1 - (distances[0] / max_dist)

    if genre and genre != "All":
        recs = recs[recs["track_genre"].str.lower() == genre.lower()]

    return recs[["track_name", "artists", "track_genre", "popularity", "match_score"]].head(n_neighbors)


# ----------------------- UI -----------------------
st.set_page_config(
    page_title="Spotify Music Recommender",
    page_icon="🎵",
    layout="wide"
)

st.title(" Spotify Music Recommendation System")
st.markdown("Compare **Baseline KNN** vs **Autoencoder Embeddings** • Mood-based & Track-based recommendations")



# Sidebar
st.sidebar.header("Settings")
mode = st.sidebar.radio("Recommendation Mode", ["Track-based", "Mood-based"])
n_neighbors = st.sidebar.slider("Number of recommendations", 5, 20, 10)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Recommendations", "Mood-based", "Visualizations", "Clustering", "Mood Profiles"]
)

with tab3:
    st.subheader("Exploratory Data Analysis")
    st.pyplot(cached_feature_distributions(df))
    st.pyplot(cached_genre_counts(df))
    st.pyplot(cached_correlation_heatmap(df))
    st.pyplot(cached_popularity_distribution(df))

    st.subheader("Autoencoder Training Loss")
    try:
        with open("outputs/autoencoder_history.pkl", "rb") as f:
            history = pickle.load(f)
        st.pyplot(cached_loss_curve(history))
    except FileNotFoundError:
        st.warning("Training history not found. Run autoencoder.py with history saving enabled.")

with tab4:
    st.subheader("Embedding Clusters")
    st.pyplot(cached_cluster_plot(embeddings, cluster_model.labels_, method="pca"))

with tab5:
    st.subheader("Mood Radar Charts")
    for mood_name, mood_vector in MOOD_MAP.items():
        st.pyplot(cached_mood_radar(mood_name, mood_vector))

# ----------------------- Track-based Mode -----------------------
if mode == "Track-based":
    st.subheader(" Track-based Recommendations")

    track_list = df["track_name"].dropna().unique().tolist()
    selected_track = st.selectbox(
        "Search or select a track",
        options=[""] + sorted(track_list),
        index=0,
        placeholder="Type to search..."
    )

    if selected_track:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("###  Baseline KNN")
            baseline_recs = get_recommendations(
                selected_track, df, baseline_knn,
                n_neighbors=n_neighbors
            )
            if baseline_recs is not None:
                st.dataframe(
                    baseline_recs.style.format({"popularity": "{:.0f}", "distance": "{:.3f}"}),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("Track not found in dataset.")

        with col2:
            st.markdown("###  Autoencoder Embeddings")
            auto_recs = get_recommendations(
                selected_track, df, auto_knn,
                n_neighbors=n_neighbors, is_autoencoder=True, _encoder=encoder
            )
            if auto_recs is not None:
                st.dataframe(
                    auto_recs.style.format({"popularity": "{:.0f}", "distance": "{:.3f}"}),
                    use_container_width=True,
                    hide_index=True
                )

        original = df[df["track_name"].str.lower() == selected_track.lower()].iloc[0]
        with st.expander("Original Track Info"):
            st.write(f"**Track:** {original['track_name']}")
            st.write(f"**Artist:** {original['artists']}")
            st.write(f"**Genre:** {original['track_genre']}")
            st.write(f"**Popularity:** {original['popularity']}")

else:
    st.subheader(" Mood-based Recommendations")

    col_m1, col_m2 = st.columns([2, 1])
    with col_m1:
        mood = st.selectbox("Select Mood", options=list(MOOD_MAP.keys()))
    with col_m2:
        genres = ["All"] + sorted(df["track_genre"].dropna().unique().tolist())
        selected_genre = st.selectbox("Filter by Genre (optional)", options=genres)

    if st.button("Generate Mood Recommendations", type="primary"):
        recs = get_mood_recommendations(
            mood, df, encoder, auto_knn, scaler,
            n_neighbors=n_neighbors, genre=selected_genre
        )

        if recs is not None and not recs.empty:
            st.success(f"Top recommendations for **{mood}** mood" +
                      (f" in **{selected_genre}**" if selected_genre != "All" else ""))
            st.dataframe(
                recs.style.format({"popularity": "{:.0f}", "distance": "{:.3f}"}),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No recommendations found for this combination.")

st.markdown("---")
st.caption("Built with Streamlit • Baseline KNN vs Autoencoder Embeddings • Spotify Tracks Dataset")