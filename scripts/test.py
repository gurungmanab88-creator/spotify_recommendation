from autoencoder import train_autoencoder
from evaluation import (
    evaluate_genre_consistency,
    evaluate_mood_alignment,
    evaluate_diversity
)

# Train or load your autoencoder + KNN
df, encoder, knn = train_autoencoder()

# Run evaluations
evaluate_genre_consistency(df, encoder, knn, "Shape of You")
evaluate_mood_alignment(df, encoder, knn, "sad")
evaluate_diversity(df, encoder, knn, "Shape of You")
