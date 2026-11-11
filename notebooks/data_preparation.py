

import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


DATA_PATH = PROJECT_DIR / "data" / "Sample-Data-CPU-Performance.csv"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "Data_Processing"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(DATA_PATH)

print("\nOriginal Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

numeric_columns = [
    'CycleTime',
    'MinimumMainMemory',
    'MaximumMainMemory',
    'CacheSize'
]

for col in numeric_columns:
    raw = df[col].astype(str)

    
    quantity_values = raw.str.extract(
        r'Quantity\[\s*([-+]?\d*\.?\d+)\s*\]',
        expand=False
    )

    plain_values = raw.str.extract(
        r'([-+]?\d*\.?\d+)',
        expand=False
    )

    df[col] = pd.to_numeric(
        quantity_values.fillna(plain_values),
        errors='coerce'
    )

print("\nMissing Values:")
print(df.isnull().sum())


print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\nShape After Removing Duplicates:")
print(df.shape)


feature_columns = [
    'CycleTime',
    'MinimumMainMemory',
    'MaximumMainMemory',
    'CacheSize',
    'MinimumNumberOfChannels',
    'MaximumNumberOfChannels'
]

target_column = 'EstimatedPerformance'

X = df[feature_columns]
y = df[target_column]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

df.to_csv(OUTPUT_DIR / "cleaned_dataset.csv", index=False)

print("\nPreprocessing Completed Successfully!")

print("Final Dataset Shape:", df.shape)
print("Final Features Shape:", X.shape)
print("Final Target Shape:", y.shape)

print("\nFiles saved in:", OUTPUT_DIR)

