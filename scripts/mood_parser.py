import numpy as np
import joblib
from config import AUDIO_FEATURES   


scaler = joblib.load("outputs/scaler.pkl")

MOOD_MAP = {
    "sad":   {"energy": 0.2, "valence": 0.2, "acousticness": 0.6},
    "happy": {"energy": 0.8, "valence": 0.9, "acousticness": 0.7},
    "party": {"energy": 0.9, "valence": 0.9, "acousticness": 0.8},
    "chill": {"energy": 0.4, "valence": 0.5, "acousticness": 0.7},
    "angry": {"energy": 0.9, "valence": 0.2, "acousticness": 0.8},
}

def mood_to_vector(mood):
    mapping = MOOD_MAP.get(mood.lower())
    if not mapping:
        print("Mood not recognized. Try: sad, happy, party, chill, angry.")
        return None

    vec = np.array([mapping.get(feat, 0.5) for feat in AUDIO_FEATURES]).reshape(1, -1)
    return scaler.transform(vec)
