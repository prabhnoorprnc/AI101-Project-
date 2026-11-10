# ==========================================
# Project: AI101 - CPU Performance Prediction
# Task: Model Training
# Member 3
# ==========================================

import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    PROJECT_DIR / "outputs" / "Data_Processing"
)

OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "Model_Training"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load features and target from Member 1
X = pd.read_csv(INPUT_DIR / "features.csv")

y_df = pd.read_csv(INPUT_DIR / "target.csv")
y = y_df.iloc[:, 0]

# Check data
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

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\nModel training completed.")

# Display coefficients
print("\nModel coefficients:")
for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, ":", coefficient)

print("\nIntercept:", model.intercept_)

# Save trained model
joblib.dump(
    model,
    OUTPUT_DIR / "linear_regression_model.pkl"
)

# Save test data for Member 4
X_test.to_csv(
    OUTPUT_DIR / "X_test.csv",
    index=False
)

y_test.to_frame(name=y.name or "EstimatedPerformance").to_csv(
    OUTPUT_DIR / "y_test.csv",
    index=False
)

# Save model coefficients
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

coefficients.to_csv(
    OUTPUT_DIR / "model_coefficients.csv",
    index=False
)

print("\nModel and test data saved successfully.")
print("Outputs saved in:", OUTPUT_DIR)
