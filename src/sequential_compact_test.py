
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis

INPUT = Path("data/processed/compact_behavioral_dataset.csv")
OUTPUT = Path("data/processed")

MIN_CHUNKS = 6
BASELINE_RATIO = 0.70

df = pd.read_csv(INPUT)

identifier_columns = ["story", "chunk", "word_count"]
feature_columns = [
    col for col in df.columns
    if col not in identifier_columns
]

results = []

for story, group in df.groupby("story"):

    group = group.sort_values("chunk")

    if len(group) < MIN_CHUNKS:
        continue

    split_index = int(len(group) * BASELINE_RATIO)
    split_index = max(2, min(split_index, len(group) - 1))

    baseline = group.iloc[:split_index]
    test = group.iloc[split_index:]

    X_baseline = baseline[feature_columns].to_numpy()
    X_test = test[feature_columns].to_numpy()

    # Standardize using reference data only
    means = X_baseline.mean(axis=0)
    stds = X_baseline.std(axis=0)
    stds[stds == 0] = 1e-8

    X_baseline = (X_baseline - means) / stds
    X_test = (X_test - means) / stds

    # Estimate regularized covariance
    model = LedoitWolf()
    model.fit(X_baseline)

    inverse_covariance = np.linalg.pinv(model.covariance_)
    center = X_baseline.mean(axis=0)

    for i, (_, row) in enumerate(test.iterrows()):

        score = mahalanobis(
            X_test[i],
            center,
            inverse_covariance
        )

        results.append({
            "story": story,
            "chunk": row["chunk"],
            "mahalanobis_score": score,
            "baseline_chunks": len(baseline),
            "test_chunks": len(test)
        })

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT / "sequential_compact_scores.csv",
    index=False
)

summary = (
    results_df
    .groupby("story")["mahalanobis_score"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .sort_values("mean", ascending=False)
)

summary.to_csv(
    OUTPUT / "sequential_compact_summary.csv",
    index=False
)

print("\nCompact sequential evaluation")
print("Stories evaluated:", results_df["story"].nunique())
print("Baseline chunks:", results_df.groupby("story")["baseline_chunks"].first().sum())
print("Test chunks:", len(results_df))
print("Features:", len(feature_columns))

print("\nStory-level results:")
print(summary.round(3).to_string(index=False))

print("\nSaved:")
print("- sequential_compact_scores.csv")
print("- sequential_compact_summary.csv")
