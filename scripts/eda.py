import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import AUDIO_FEATURES

def plot_feature_distributions(df):
    fig = plt.figure(figsize=(15,10))
    df[AUDIO_FEATURES].hist(bins=30, figsize=(15,10))
    plt.suptitle("Distribution of Audio Features", fontsize=16)
    return fig

def plot_genre_counts(df):
    fig = plt.figure(figsize=(12,6))
    df["track_genre"].value_counts().plot(kind="bar")
    plt.title("Track Count per Genre")
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df):
    fig = plt.figure(figsize=(10,8))
    corr = df[AUDIO_FEATURES].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Audio Features")
    return fig

def plot_popularity_distribution(df):
    fig = plt.figure(figsize=(8,5))
    sns.histplot(df["popularity"], bins=30, kde=True)
    plt.title("Popularity Distribution")
    plt.xlabel("Popularity (0–100)")
    plt.ylabel("Frequency")
    return fig
