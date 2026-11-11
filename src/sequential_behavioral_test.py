
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

MIN_CHUNKS = 4
BASELINE_RATIO = 0.70

# --------------------------------
# Load data
# --------------------------------

df = pd.read_csv(INPUT)

identifier_columns = ["story", "chunk", "word_count"]
feature_columns = [
    col for col in df.columns
    if col not in identifier_columns
]

# --------------------------------
# Split each story sequentially
# --------------------------------

baseline_chunks = []
test_chunks = []

for story, group in df.groupby("story"):

    # Preserve original chunk order
    group = group.sort_values("chunk")

    if len(group) < MIN_CHUNKS:
        continue

    split_index = int(len(group) * BASELINE_RATIO)

    # Ensure both sections contain at least one chunk
    split_index = max(1, min(split_index, len(group) - 1))

    baseline_chunks.append(group.iloc[:split_index])
    test_chunks.append(group.iloc[split_index:])

baseline = pd.concat(baseline_chunks)
test = pd.concat(test_chunks)

# --------------------------------
# Build reference profile
# --------------------------------

X_baseline = baseline[feature_columns].to_numpy()
X_test = test[feature_columns].to_numpy()

# Standardize using baseline only
means = X_baseline.mean(axis=0)
stds = X_baseline.std(axis=0)

stds[stds == 0] = 1e-8

X_baseline = (X_baseline - means) / stds
X_test = (X_test - means) / stds

# Estimate covariance
model = LedoitWolf()
model.fit(X_baseline)

inverse_covariance = np.linalg.pinv(model.covariance_)
center = X_baseline.mean(axis=0)

# --------------------------------
# Calculate anomaly scores
# --------------------------------

results = []

for i, (_, row) in enumerate(test.iterrows()):

    score = mahalanobis(
        X_test[i],
        center,
        inverse_covariance
    )

    results.append({
        "story": row["story"],
        "chunk": row["chunk"],
        "mahalanobis_score": score
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT / "sequential_anomaly_scores.csv",
    index=False
)

# --------------------------------
# Summarize results
# --------------------------------

summary = (
    results_df
    .groupby("story")["mahalanobis_score"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .sort_values("mean", ascending=False)
)

summary.to_csv(
    OUTPUT / "sequential_anomaly_summary.csv",
    index=False
)

print("\nSequential evaluation:")
print("Stories included:", baseline["story"].nunique())
print("Baseline chunks:", len(baseline))
print("Test chunks:", len(test))

print("\nStory-level results:")
print(summary.round(3).to_string(index=False))

print("\nGenerated:")
print("- sequential_anomaly_scores.csv")
print("- sequential_anomaly_summary.csv")
