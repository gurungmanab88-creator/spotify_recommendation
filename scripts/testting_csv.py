import pandas as pd
import dataframe_image as dfi

# Load dataset
df = pd.read_csv("tracks.csv")  # replace with your file

# Select a compact sample to screenshot
sample = df.head(12)  # adjust rows shown

# Optional: reorder or select columns to display
cols = ["track_name","artist_name","genre","danceability","energy","valence","tempo"]
sample = sample[cols]

# Save styled dataframe as PNG
styled = sample.style.set_table_styles(
    [{"selector":"th","props":[("background-color","#f2f2f2"),("font-weight","bold")]}]
).format({"tempo":"{:.0f}","danceability":"{:.2f}","energy":"{:.2f}","valence":"{:.2f}"})

dfi.export(styled, "dataset_snapshot.png")
print("Saved dataset_snapshot.png")