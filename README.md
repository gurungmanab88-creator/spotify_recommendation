Spotify Music Recommendation System
An end-to-end machine learning pipeline for personalized music recommendation. The system combines a baseline K-Nearest Neighbors (KNN) model, autoencoder-based embeddings, clustering (KMeans/GMM), and a Streamlit web interface for interactive exploration.
The project covers the full workflow: data preprocessing, model training, evaluation, and deployment of a user-friendly recommendation UI.

PRoject Structure
├── data/                  # Raw dataset (spotify_tracks.csv)
├── outputs/               # Saved artifacts (scaler, models, embeddings, figures)
├── config.py              # Global configuration (paths, AUDIO_FEATURES)
├── preprocess.py          # Data cleaning and feature scaling
├── baseline.py            # Baseline KNN recommender
├── autoencoder.py         # Autoencoder training and embedding generation
├── clustering.py          # KMeans and GMM clustering
├── eda.py                 # Exploratory data analysis
├── evaluation.py          # Recommendation metrics (genre consistency, mood alignment, diversity)
├── mood_parser.py         # Mood-to-feature vector mapping (scaled)
├── retrieval.py           # Core recommendation functions
├── app.py                 # Streamlit interactive UI
└── requirements.txt       # Project dependencies

Setup
clone repo =  git clone https://github.com/yourusername/music-recommender.git
              cd music-recommender

install dependencies = pip install -r requirements.txt

Features
Baseline vs Autoencoder recommendations side by side

Mood‑based retrieval (sad, happy, party, chill, angry, romantic, focus, sleepy, motivated, calm, epic, melancholy, uplifting, dark, energetic)

Cluster analysis (KMeans/GMM)

Evaluation metrics: genre consistency, mood alignment, diversity

Streamlit UI for interactive exploration
