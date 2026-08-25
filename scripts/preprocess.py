import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess(
    input_path = "data/spotify_tracks.csv",
    cleaned_path = "outputs/cleaned_tracks.csv",
    scaled_path = "outputs/scaled_features.csv",
    scaler_path = "outputs/scaler.pkl"
):
   os.makedirs("outputs", exist_ok = True)

   #loading the raw dataset 

   df = pd.read_csv(input_path)

   #dropping the stray index coloumn i learned through reddit

   df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

   #using these audio features

   audio_features =[
    "danceability", "energy", "valence", "tempo", 
    "acousticness", "instrumentalness", "liveness", 
    "speechiness", "loudness"
   ]

   #dropping rows where there might be missing track name, artists or otheres features
   df = df.dropna(subset=["track_name", "artists"] + audio_features)

    # keeping the hihest pouplarity per track name and artuist 

   df = df.sort_values("popularity", ascending = False)
   df = df.drop_duplicates(subset=["track_name", "artists"], keep = "first")

    #saving the cleaned dataset 
   df.to_csv(cleaned_path, index = False)
   print(f"cleaned daataset saved {cleaned_path}")

    #now i am going to scale the features

   scaler = StandardScaler()
   scaled_features = scaler.fit_transform(df[audio_features])

    # craaeting a scaled DAtaFrame with the same index

   scaled_df = pd.DataFrame(scaled_features, columns = audio_features, index=df.index)


    # keeping the track name and arstist alon scaled features

   scaled_df["track_name" ] = df["track_name"]
   scaled_df["artists" ] = df["artists"]
   scaled_df["popularity" ] = df["popularity"]
   scaled_df["track_genre" ] = df["track_genre"]

    #saving the scaled daraset

   scaled_df.to_csv(scaled_path, index = False)
   print(f"saved scaled dataset to {scaled_path}")

    #saving the scaler inorder for future use in modeltraing etc

   joblib.dump(scaler, scaler_path)
   print(f"saved scaler object to {scaler_path}")

if __name__ == "__main__":
    preprocess()
    
