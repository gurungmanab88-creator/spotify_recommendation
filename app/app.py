import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import pickle
from scripts.config import AUDIO_FEATURES, OUTPUT_DIR
# Models and data are loaded below using caching functions
encoder = load_model(os.path.join(OUTPUT_DIR, "encoder.keras"))
auto_knn = joblib.load(os.path.join(OUTPUT_DIR, "autoencoder_knn.pkl"))
baseline_knn = joblib.load(os.path.join(OUTPUT_DIR, "baseline_knn.pkl"))
scaler = joblib.load(os.path.join(OUTPUT_DIR, "scaler.pkl"))
embeddings = np.load(os.path.join(OUTPUT_DIR, "embeddings.npy"))
cluster_model = joblib.load(os.path.join(OUTPUT_DIR, "kmeans.pkl"))
df = pd.read_csv(os.path.join(OUTPUT_DIR, "cleaned_data.csv"))



# i can see the figures here

df = pd.read_csv("outputs/cleaned_data.csv")

AUDIO_FEATURES = [
    "danceability","energy","valence","tempo","acousticness",
    "instrumentalness","liveness","speechiness","loudness"
]

# Tabs
tab1, tab2, tab3 = st.tabs(["Recommendations", "Mood-based", "Visualizations"])

with tab3:
    st.subheader("Exploratory Data Analysis")

    # Feature distributions
    fig, ax = plt.subplots(figsize=(15,10))
    df[AUDIO_FEATURES].hist(bins=30, figsize=(15,10))
    plt.suptitle("Distribution of Audio Features", fontsize=16)
    st.pyplot(fig)

    # Genre counts
    fig, ax = plt.subplots(figsize=(12,6))
    df["track_genre"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Track Count per Genre")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(10,8))
    corr = df[AUDIO_FEATURES].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap of Audio Features")
    st.pyplot(fig)

    # Popularity distribution
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(df["popularity"], bins=30, kde=True, ax=ax)
    ax.set_title("Popularity Distribution")
    ax.set_xlabel("Popularity (0–100)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.subheader("Autoencoder Training Loss")
    try:
        with open("outputs/autoencoder_history.pkl", "rb") as f:
            history = pickle.load(f)

        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(history["loss"], label="Training Loss")
        if "val_loss" in history:
            ax.plot(history["val_loss"], label="Validation Loss")
        ax.set_title("Autoencoder Training Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        st.pyplot(fig)

    except FileNotFoundError:
        st.warning("Training history not found. Run autoencoder.py with history saving enabled.")

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
# ----------------------- Caching -----------------------
@st.cache_resource
def load_models():
    encoder = load_model(os.path.join(OUTPUT_DIR, "encoder.h5"))
    auto_knn = joblib.load(os.path.join(OUTPUT_DIR, "autoencoder_knn.pkl"))
    baseline_knn = joblib.load(os.path.join(OUTPUT_DIR, "baseline_knn.pkl"))
    return encoder, auto_knn, baseline_knn

@st.cache_data
def load_data():
    scaler = joblib.load(os.path.join(OUTPUT_DIR, "scaler.pkl"))
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "scaled_features.csv"))
    return df, scaler

# ----------------------- Helper Functions -----------------------
def mood_to_vector(mood: str, scaler):
    mapping = MOOD_MAP.get(mood.lower())
    if mapping is None:
        return None
    vec = np.array([mapping.get(feat, 0.5) for feat in AUDIO_FEATURES]).reshape(1, -1)
    return scaler.transform(vec)

def get_recommendations(track_name, df, knn, n_neighbors=10, is_autoencoder=False, encoder=None):
    track = df[df["track_name"].str.lower() == track_name.lower()]
    if track.empty:
        return None

    idx = track.index[0]
    X = df[AUDIO_FEATURES].values

    if is_autoencoder:
        embedding = encoder.predict(X[idx].reshape(1, -1), verbose=0)
        distances, indices = knn.kneighbors(embedding, n_neighbors=n_neighbors)
    else:
        distances, indices = knn.kneighbors([X[idx]], n_neighbors=n_neighbors)

    recs = df.iloc[indices[0]].copy()
    recs["distance"] = distances[0]
    return recs[["track_name", "artists", "track_genre", "popularity", "distance"]]


def get_mood_recommendations(mood, df, encoder, knn, scaler, n_neighbors=10, genre=None):
    vec = mood_to_vector(mood, scaler)
    if vec is None:
        return None

    embedding = encoder.predict(vec, verbose=0)
    distances, indices = knn.kneighbors(embedding, n_neighbors=n_neighbors * 3)  # get more then filter
    recs = df.iloc[indices[0]].copy()
    recs["distance"] = distances[0]

    if genre and genre != "All":
        recs = recs[recs["track_genre"].str.lower() == genre.lower()]

    return recs[["track_name", "artists", "track_genre", "popularity", "distance"]].head(n_neighbors)

# ----------------------- UI -----------------------
st.set_page_config(
    page_title="Spotify Music Recommender",
    page_icon="🎵",
    layout="wide"
)

st.title(" Spotify Music Recommendation System")
st.markdown("Compare **Baseline KNN** vs **Autoencoder Embeddings** • Mood-based & Track-based recommendations")

# Load everything
encoder, auto_knn, baseline_knn = load_models()
df, scaler = load_data()


# Sidebar
st.sidebar.header("Settings")
mode = st.sidebar.radio("Recommendation Mode", ["Track-based", "Mood-based"])

n_neighbors = st.sidebar.slider("Number of recommendations", 5, 20, 10)

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
                n_neighbors=n_neighbors, is_autoencoder=True, encoder=encoder
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