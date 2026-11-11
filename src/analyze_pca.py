
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

# Paths
INPUT = Path("data/processed/standardized_behavioral_dataset.csv")
OUTPUT = Path("data/processed")
PLOT_DIR = OUTPUT / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Load standardized dataset
df = pd.read_csv(INPUT)

# Keep identifiers separate from PCA
identifiers = df[["story", "chunk", "word_count"]]

# Select the 14 standardized features
features = df.drop(columns=["story", "chunk", "word_count"])

# Perform PCA
pca = PCA()
scores = pca.fit_transform(features)

# --------------------------------
# 1. Explained variance
# --------------------------------

variance_df = pd.DataFrame({
    "principal_component": [
        f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))
    ],
    "explained_variance": pca.explained_variance_ratio_,
    "cumulative_variance": pca.explained_variance_ratio_.cumsum()
})

variance_df.to_csv(
    OUTPUT / "pca_explained_variance.csv",
    index=False
)

print("\nExplained variance:")
print(variance_df.round(4))

# --------------------------------
# 2. PCA scores
# --------------------------------

scores_df = pd.DataFrame(
    scores,
    columns=[f"PC{i + 1}" for i in range(scores.shape[1])]
)

scores_df = pd.concat(
    [identifiers, scores_df],
    axis=1
)

scores_df.to_csv(
    OUTPUT / "pca_chunk_scores.csv",
    index=False
)

# --------------------------------
# 3. Feature loadings
# --------------------------------

loadings_df = pd.DataFrame(
    pca.components_.T,
    index=features.columns,
    columns=[f"PC{i + 1}" for i in range(len(features.columns))]
)

loadings_df.to_csv(
    OUTPUT / "pca_feature_loadings.csv"
)

print("\nFeature loadings for PC1 and PC2:")
print(loadings_df[["PC1", "PC2"]].round(3))

# --------------------------------
# 4. Explained variance plot
# --------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, len(variance_df) + 1),
    variance_df["cumulative_variance"],
    marker="o"
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA: Cumulative Explained Variance")
plt.xticks(range(1, len(variance_df) + 1))
plt.grid(True)
plt.tight_layout()

plt.savefig(PLOT_DIR / "pca_explained_variance.png")
plt.close()

# --------------------------------
# 5. PC1 vs PC2 scatter plot
# --------------------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    scores[:, 0],
    scores[:, 1],
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: Behavioral Representation of Writing Chunks")
plt.grid(True)
plt.tight_layout()

plt.savefig(PLOT_DIR / "pca_chunks.png")
plt.close()

print("\nPCA analysis complete!")
print("Generated:")
print("- pca_explained_variance.csv")
print("- pca_chunk_scores.csv")
print("- pca_feature_loadings.csv")
print("- pca_explained_variance.png")
print("- pca_chunks.png")
