Spotify Music Recommendation System
An end-to-end machine learning pipeline for personalized music recommendation. The system combines a baseline K-Nearest Neighbors (KNN) model, autoencoder-based embeddings, clustering (KMeans/GMM), and a Streamlit web interface for interactive exploration.
The project covers the full workflow: data preprocessing, model training, evaluation, and deployment of a user-friendly recommendation UI.

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

Future Work 

Integrate cluster‑based recommendations into Streamlit
Expand mood mapping with dataset‑driven averages
Add automated tests + CI/CD
Improve README with screenshots of the app
