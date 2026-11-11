
import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------
# Configuration
# --------------------------------

INPUT = Path("data/processed/reduced_behavioral_dataset.csv")

# --------------------------------
# Load dataset
# --------------------------------

df = pd.read_csv(INPUT)

identifier_columns = ["story", "chunk", "word_count"]
feature_columns = [
    column for column in df.columns
    if column not in identifier_columns
]

stories = df["story"].unique()

# --------------------------------
# Inspect each held-out story
# --------------------------------

all_contributions = []

for held_out_story in stories:

    reference = df[df["story"] != held_out_story]
    test = df[df["story"] == held_out_story]

    means = reference[feature_columns].mean()
    stds = reference[feature_columns].std(ddof=0)
    stds = stds.replace(0, 1e-8)

    test_z = (test[feature_columns] - means) / stds

    for index, row in test.iterrows():

        # Absolute deviation per feature
        deviations = test_z.loc[index].abs()

        for feature, deviation in deviations.items():
            all_contributions.append({
                "story": held_out_story,
                "chunk": row["chunk"],
                "feature": feature,
                "absolute_z_score": deviation
            })

# --------------------------------
# Save feature contributions
# --------------------------------

contributions = pd.DataFrame(all_contributions)

contributions.to_csv(
    "data/processed/anomaly_feature_contributions.csv",
    index=False
)

# --------------------------------
# Display largest deviations
# --------------------------------

top_contributions = (
    contributions
    .sort_values("absolute_z_score", ascending=False)
    .groupby(["story", "chunk"])
    .head(3)
)

print("\nTop 3 contributing features per chunk:\n")

for (story, chunk), group in top_contributions.groupby(["story", "chunk"]):

    print(f"\n{story} — {chunk}")

    for _, row in group.iterrows():
        print(
            f"  {row['feature']}: "
            f"{row['absolute_z_score']:.3f}"
        )

print("\nSaved to:")
print("data/processed/anomaly_feature_contributions.csv")
