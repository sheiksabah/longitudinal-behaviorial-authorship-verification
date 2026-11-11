
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis

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

stories = df["story"].unique()

results = []

# --------------------------------
# Leave-one-story-out evaluation
# --------------------------------

for held_out_story in stories:

    reference = df[df["story"] != held_out_story]
    test = df[df["story"] == held_out_story]

    X_reference = reference[feature_columns].to_numpy()
    X_test = test[feature_columns].to_numpy()

    # Standardize using reference data only
    means = X_reference.mean(axis=0)
    stds = X_reference.std(axis=0)

    stds[stds == 0] = 1e-8

    X_reference = (X_reference - means) / stds
    X_test = (X_test - means) / stds

    # Estimate regularized covariance
    covariance_model = LedoitWolf()
    covariance_model.fit(X_reference)

    covariance = covariance_model.covariance_
    inverse_covariance = np.linalg.pinv(covariance)

    # Reference center
    center = X_reference.mean(axis=0)

    # Calculate Mahalanobis distance
    for i, row in enumerate(X_test):

        score = mahalanobis(
            row,
            center,
            inverse_covariance
        )

        results.append({
            "story": held_out_story,
            "chunk": test.iloc[i]["chunk"],
            "mahalanobis_score": score
        })

# --------------------------------
# Save chunk-level scores
# --------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT / "mahalanobis_anomaly_scores.csv",
    index=False
)

# --------------------------------
# Summarize by story
# --------------------------------

summary = (
    results_df
    .groupby("story")["mahalanobis_score"]
    .agg(["mean", "std", "min", "max", "count"])
    .reset_index()
    .sort_values("mean", ascending=False)
)

summary.to_csv(
    OUTPUT / "mahalanobis_anomaly_summary.csv",
    index=False
)

print("\nMahalanobis anomaly scores:")
print(summary.round(3).to_string(index=False))

print("\nAnalysis complete!")
print("Stories evaluated:", len(stories))
print("Chunks evaluated:", len(results_df))
print("Features used:", len(feature_columns))
