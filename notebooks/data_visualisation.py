
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_DIR / "outputs" / "cleaned_dataset.csv"

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load cleaned dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())


# --------------------------------------------------
# 3. Check missing values
# --------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())


# --------------------------------------------------
# 4. Check duplicate rows
# --------------------------------------------------

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())


# --------------------------------------------------
# 5. Correlation analysis
# --------------------------------------------------

correlation = df.corr(numeric_only=True)

print("\nCorrelation matrix:")
print(correlation)


# --------------------------------------------------
# 6. Correlation heatmap
# --------------------------------------------------

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 7. Feature vs Target relationships
# --------------------------------------------------

target_column = "EstimatedPerformance"

feature_columns = [
    "CycleTime",
    "MinimumMainMemory",
    "MaximumMainMemory",
    "CacheSize",
    "MinimumNumberOfChannels",
    "MaximumNumberOfChannels"
]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

axes = axes.flatten()

for i, column in enumerate(feature_columns):

    sns.scatterplot(
        data=df,
        x=column,
        y=target_column,
        ax=axes[i],
        alpha=0.7
    )

    axes[i].set_title(f"{column} vs {target_column}")

fig.suptitle(
    "Feature Relationships with Estimated Performance",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_relationships.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 8. Completion message
# --------------------------------------------------

print("\nData analysis and visualization completed!")
print("Graphs saved in:", OUTPUT_DIR)
