# Project: AI101 - CPU Performance Prediction
# Task: Model Training
# Member 3

import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# 1. Project paths

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_DIR / "outputs" / "cleaned_dataset.csv"

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



# 2. Load cleaned dataset from Member 1


df = pd.read_csv(INPUT_PATH)

print("\nCleaned dataset loaded successfully!")

print("\nDataset shape:")
print(df.shape)



# 3. Separate features and target


feature_columns = [
    "CycleTime",
    "MinimumMainMemory",
    "MaximumMainMemory",
    "CacheSize",
    "MinimumNumberOfChannels",
    "MaximumNumberOfChannels"
]

X = df[feature_columns]
y = df["EstimatedPerformance"]



# 4. Data Checking


print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)


# Check for missing values
if X.isnull().any().any() or y.isnull().any():
    raise ValueError(
        "Missing values found. Please handle them in data preprocessing."
    )


# --------------------------------------------------
# 5. Split dataset into training and testing sets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# 6. Create Linear Regression model

model = LinearRegression()



# 7. Model Training

model.fit(X_train, y_train)

print("\nModel training completed successfully!")


# 8. Display model coefficients

print("\nModel coefficients:")

for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, ":", coefficient)

print("\nIntercept:", model.intercept_)


joblib.dump(
    model,
    OUTPUT_DIR / "trained_model.pkl"
)

print("\nTrained model saved successfully!")



# 10. Completion message

print("\nModel training completed successfully!")
print("Model saved in:", OUTPUT_DIR)
