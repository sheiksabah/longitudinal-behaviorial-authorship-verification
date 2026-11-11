
import pandas as pd
import numpy as np

from pathlib import Path

INPUT = Path("data/processed/mahalanobis_anomaly_scores.csv")
OUTPUT = Path("data/processed")

# Load leave-one-story-out anomaly scores
df = pd.read_csv(INPUT)

# Calculate maximum score per story
story_scores = (
    df.groupby("story")["mahalanobis_score"]
    .max()
    .reset_index(name="maximum_score")
)

# Calculate provisional threshold
threshold = np.quantile(
    story_scores["maximum_score"],
    0.90,
    method="higher"
)

# Identify stories exceeding threshold
story_scores["flagged"] = (
    story_scores["maximum_score"] > threshold
)

# Save results
story_scores.to_csv(
    OUTPUT / "threshold_calibration.csv",
    index=False
)

print("\nThreshold calibration")
print("---------------------")
print("Stories:", len(story_scores))
print("90th percentile threshold:", round(threshold, 4))

print(
    "Stories above threshold:",
    story_scores["flagged"].sum()
)

print("\nStory-level results:")
print(
    story_scores.sort_values(
        "maximum_score",
        ascending=False
    ).round(3).to_string(index=False)
)

print("\nSaved: data/processed/threshold_calibration.csv")
