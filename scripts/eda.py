import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.config import AUDIO_FEATURES, OUTPUT_DIR


def plot_feature_distributions(df):
    
    fig, axes = plt.subplots(3, 3, figsize=(15,10))
    for i, feat in enumerate(AUDIO_FEATURES):
        ax = axes[i//3, i%3]
        sns.histplot(df[feat], bins=30, kde=True, ax=ax, color="skyblue")
        ax.set_title(feat.capitalize())
    plt.suptitle("Distribution of Audio Features", fontsize=16)
    plt.tight_layout()
    return fig

def plot_genre_counts(df, top_n=15):
  
    genre_counts = df["track_genre"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(y=genre_counts.index, x=genre_counts.values, palette="viridis", ax=ax)
    ax.set_title(f"Top {top_n} Genres by Track Count")
    ax.set_xlabel("Count")
    ax.set_ylabel("Genre")
  
    for i, v in enumerate(genre_counts.values):
        ax.text(v + 5, i, str(v), color="black", va="center")
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df):
  
    fig, ax = plt.subplots(figsize=(10,8))
    corr = df[AUDIO_FEATURES].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap of Audio Features")
    return fig

def plot_popularity_distribution(df):
    
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(df["popularity"], bins=30, kde=True, color="steelblue", ax=ax)
    mean_val = df["popularity"].mean()
    median_val = df["popularity"].median()
    ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.2f}")
    ax.axvline(median_val, color="green", linestyle="--", label=f"Median: {median_val:.2f}")
    ax.set_title("Popularity Distribution")
    ax.set_xlabel("Popularity (0–100)")
    ax.set_ylabel("Frequency")
    ax.legend()
    return fig
