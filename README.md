#Spotify Music Recommendation System

An end‑to‑end machine learning project that recommends songs based on track similarity and user moods.  
Built with **Python, Scikit‑Learn, TensorFlow/Keras, and Streamlit**.

---

# Features
- **Baseline Recommendations**: KNN on raw audio features.
- **Autoencoder Recommendations**: Latent embeddings for deeper similarity.
- **Mood‑Based Playlists**: Generate recommendations for moods (happy, sad, party, chill).
- **Clustering**: KMeans/GMM grouping with t‑SNE/PCA visualization.
- **EDA Visualizations**: Feature distributions, genre counts, correlation heatmap, popularity distribution.
- **Interactive Streamlit App**: Tabbed interface with recommendations and plots.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/gurungmanab88-creator/spotify_recommendation.git
cd spotify_recommendation
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt

run the pipeline
python scripts/preprocess.py
python scripts/autoencoder.py
python scripts/clustering.py
python scripts/save_model.py
streamlit run app/app.py






