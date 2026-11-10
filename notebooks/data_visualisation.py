import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_DIR / "outputs" / "Data_Processing" / "cleaned_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "Data_Analysis_Visualization"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned dataset from Member 1
df = pd.read_csv(INPUT_PATH)

# Display basic information
print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

# Check missing values
missing_values = df.isnull().sum()

print("\nMissing values:")
print(missing_values)

missing_values.to_csv(OUTPUT_DIR / "missing_values.csv")

# Check duplicate rows
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# Save statistical summary
df.describe().to_csv(OUTPUT_DIR / "summary_statistics.csv")

# Correlation analysis
correlation = df.corr(numeric_only=True)

print("\nCorrelation matrix:")
print(correlation)

correlation.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

# Correlation heatmap
plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
plt.close()

# Histograms of numerical columns
df.hist(figsize=(12, 9), bins=20)

plt.suptitle("Distribution of CPU Dataset Features")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "feature_distributions.png")
plt.close()

# Feature vs target scatter plots
target_column = "EstimatedPerformance"

feature_columns = [
    "CycleTime",
    "MinimumMainMemory",
    "MaximumMainMemory",
    "CacheSize",
    "MinimumNumberOfChannels",
    "MaximumNumberOfChannels"
]

for column in feature_columns:

    plt.figure(figsize=(7, 5))

    sns.scatterplot(
        data=df,
        x=column,
        y=target_column
    )

    plt.title(f"{column} vs {target_column}")
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / f"{column}_vs_performance.png")
    plt.close()

print("\nData analysis and visualization completed.")
print("Outputs saved in:", OUTPUT_DIR)
