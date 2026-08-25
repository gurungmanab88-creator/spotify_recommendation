import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import os

def run_eda(input_path = "outputs/cleaned_tracks.csv", output_dir = "outputs/figures/"):

    os.makedirs(output_dir, exist_ok =True)

    df = pd.read_csv(input_path)

    audio_features = [
        "danceability","energy","valence","tempo","acousticness","instrumentalness","liveness","speechiness","loudness"
        ]

    #Histogeam of above features
    df[audio_features].hist(bins = 30, figsize=(15,10))
    plt.suptitle("Distribution audio feature", fontsize = 16)
    plt.savefig(f"{output_dir}/audio_features_hist.png")
    plt.close()

    #genre bar chart
    plt.figure(figsize = (12,6))
    df["track_genre"].value_counts().plot(kind = "bar")
    plt.title("TRack count per genre")
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/genre_counts.png")
    plt.close()

    #heatmap corealtion
    plt.figure(figsize = (10,8))
    corr = df[audio_features].corr()
    sns.heatmap(corr, annot = True, cmap = "coolwarm", fmt = ".2f")
    plt.title("Correaltion heatmap of audio_features")
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()

    #Popularity distibution
    plt.figure(figsize = (8,5))
    sns.histplot(df["popularity"], bins=30, kde=True)
    plt.title("popularity distribution")
    plt.xlabel("Popularity(0-100)")
    plt.ylabel('Frequnecy')
    plt.savefig(f"{output_dir}/popularity_distribution.png")
    plt.close()

    print(f"Eda figures saved to {output_dir}")


if __name__== "__main__":
    run_eda()


        

    