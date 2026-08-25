from baseline import AUDIO_FEATURES
import numpy as np

AUDIO_FEATURES = [
    "danceability", "energy", "valence", "tempo", 
    "acousticness", "instrumentalness", "liveness", 
    "speechiness", "loudness"
]

# for now simple mood to feature mapping

MOOD_MAP = {
    "sad": {"energy": 0.2, "valence": 0.2 , "acousticness":0.6},
    "happy": {"energy": 0.8, "valence": 0.9 , "acousticness":0.7},
    "party": {"energy": 0.9, "valence": 0.9 , "acousticness":0.8},
    "chill": {"energy": 0.4, "valence": 0.5 , "acousticness":0.7},
    "angry": {"energy": 0.9, "valence": 0.2 , "acousticness":0.8},
}


def mood_to_vector(mood):
    vec = np.zeros(len(AUDIO_FEATURES))
    mapping = MOOD_MAP.get(mood.lower())
    if not mapping:
        print("Mood not recognized. Try: sad, happy, party, chill, angry.")
        return None

  
    for i, feat in enumerate(AUDIO_FEATURES):
        vec[i] = mapping.get(feat, 0.5)
    return vec.reshape(1, -1)
    