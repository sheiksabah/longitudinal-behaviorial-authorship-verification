
import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------
# Configuration
# --------------------------------

INPUT = Path("data/processed/reduced_behavioral_dataset.csv")
OUTPUT = Path("data/processed")

# --------------------------------
# Load data
# --------------------------------

df = pd.read_csv(INPUT)

identifier_columns = ["story", "chunk", "word_count"]
feature_columns = [
    column for column in df.columns
    if column not in identifier_columns
]

results = []

# --------------------------------
# Leave-one-story-out evaluation
# --------------------------------

stories = df["story"].unique()

for held_out_story in stories:

    # Separate reference and test data by story
    reference = df[df["story"] != held_out_story]
    test = df[df["story"] == held_out_story]

    X_reference = reference[feature_columns]
    X_test = test[feature_columns]

    # Calculate reference statistics
    means = X_reference.mean()
    stds = X_reference.std(ddof=0)

    # Avoid division by zero
    stds = stds.replace(0, 1e-8)

    # Standardize using reference data only
    reference_z = (X_reference - means) / stds
    test_z = (X_test - means) / stds

    # Calculate average absolute deviation
    anomaly_scores = np.sqrt((test_z ** 2).mean(axis=1))

    # Store results
    for index, score in anomaly_scores.items():
        results.append({
            "story": held_out_story,
            "chunk": df.loc[index, "chunk"],
            "anomaly_score": score
        })

# --------------------------------
# Save chunk-level results
# --------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT / "behavioral_anomaly_scores.csv",
    index=False
)

# --------------------------------
# Summarize results by story
# --------------------------------

summary = (
    results_df
    .groupby("story")["anomaly_score"]
    .agg(["mean", "std", "min", "max", "count"])
    .reset_index()
    .sort_values("mean", ascending=False)
)

summary.to_csv(
    OUTPUT / "behavioral_anomaly_summary.csv",
    index=False
)

print("\nStory-level anomaly scores:")
print(summary.round(3).to_string(index=False))

print("\nAnalysis complete!")
print("Stories evaluated:", len(stories))
print("Chunks evaluated:", len(results_df))
print("Features used:", len(feature_columns))
print("\nGenerated:")
print("- data/processed/behavioral_anomaly_scores.csv")
print("- data/processed/behavioral_anomaly_summary.csv")
