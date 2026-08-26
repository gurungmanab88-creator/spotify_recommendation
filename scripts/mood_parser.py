import numpy as np
import joblib
from config import AUDIO_FEATURES   

scaler = joblib.load("outputs/scaler.pkl")

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

def mood_to_vector(mood):
    mapping = MOOD_MAP.get(mood.lower())
    if not mapping:
        print("Mood not recognized. Try one of:", list(MOOD_MAP.keys()))
        return None

    vec = np.array([mapping.get(feat, 0.5) for feat in AUDIO_FEATURES]).reshape(1, -1)
    return scaler.transform(vec)
