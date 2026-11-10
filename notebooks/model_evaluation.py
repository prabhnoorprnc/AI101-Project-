import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Set project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "Model_Training"

DATA_PATH = PROJECT_DIR / "outputs" / "Data_Processing" / "cleaned_dataset.csv"
MODEL_PATH = OUTPUT_DIR / "linear_regression_model.pkl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load cleaned dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Cleaned Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)


# --------------------------------------------------
# 3. Separate features and target
# --------------------------------------------------

features = [
    "CycleTime",
    "MinimumMainMemory",
    "MaximumMainMemory",
    "CacheSize",
    "MinimumNumberOfChannels",
    "MaximumNumberOfChannels"
]

X = df[features]
y = df["EstimatedPerformance"]


# --------------------------------------------------
# 4. Split dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------------------------------
# 5. Load trained model
# --------------------------------------------------

model = joblib.load(MODEL_PATH)

print("\nTrained Model Loaded Successfully!")


# --------------------------------------------------
# 6. Make predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 7. Calculate evaluation metrics
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


# --------------------------------------------------
# 8. Display evaluation results
# --------------------------------------------------

print("\nMODEL EVALUATION RESULTS")

print("----------------------------")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)


# --------------------------------------------------
# 9. Save evaluation metrics
# --------------------------------------------------

metrics = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2 Score"],
    "Value": [mae, mse, rmse, r2]
})

metrics.to_csv(
    OUTPUT_DIR / "evaluation_metrics.csv",
    index=False
)


# --------------------------------------------------
# 10. Actual vs Predicted graph
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.7)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Performance")
plt.ylabel("Predicted Performance")
plt.title("Actual vs Predicted Performance")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 11. Completion message
# --------------------------------------------------

print("\nEvaluation Completed Successfully!")

print("Evaluation metrics and graph saved in outputs folder.")
