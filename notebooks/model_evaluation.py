
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.linear_model import LinearRegression
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
# 5. Perform 5-Fold Cross-Validation
# --------------------------------------------------

print("\nPERFORMING 5-FOLD CROSS-VALIDATION")

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = cross_validate(
    estimator=LinearRegression(),
    X=X_train,
    y=y_train,
    cv=kf,
    scoring={
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2"
    }
)

# Convert negative MAE and RMSE scores into positive values
cv_mae = -cv_results["test_MAE"]
cv_rmse = -cv_results["test_RMSE"]
cv_r2 = cv_results["test_R2"]

print("\nCROSS-VALIDATION RESULTS")

for fold in range(5):
    print(
        f"Fold {fold + 1}: "
        f"MAE = {cv_mae[fold]:.2f}, "
        f"RMSE = {cv_rmse[fold]:.2f}, "
        f"R2 = {cv_r2[fold]:.4f}"
    )

print("\nAVERAGE CROSS-VALIDATION SCORES")
print("Mean MAE:", cv_mae.mean())
print("Mean RMSE:", cv_rmse.mean())
print("Mean R2:", cv_r2.mean())

print("\nSTANDARD DEVIATION")
print("MAE:", cv_mae.std())
print("RMSE:", cv_rmse.std())
print("R2:", cv_r2.std())

# Save fold-wise results
cv_metrics = pd.DataFrame({
    "Fold": [f"Fold {i}" for i in range(1, 6)],
    "MAE": cv_mae,
    "RMSE": cv_rmse,
    "R2": cv_r2
})

# Add mean and standard deviation rows
cv_summary = pd.DataFrame([
    {
        "Fold": "Mean",
        "MAE": cv_mae.mean(),
        "RMSE": cv_rmse.mean(),
        "R2": cv_r2.mean()
    },
    {
        "Fold": "Standard Deviation",
        "MAE": cv_mae.std(),
        "RMSE": cv_rmse.std(),
        "R2": cv_r2.std()
    }
])

cv_metrics = pd.concat(
    [cv_metrics, cv_summary],
    ignore_index=True
)

cv_metrics.to_csv(
    OUTPUT_DIR / "cross_validation_results.csv",
    index=False
)


# --------------------------------------------------
# 6. Load trained model
# --------------------------------------------------

model = joblib.load(MODEL_PATH)

print("\nTrained Model Loaded Successfully!")


# --------------------------------------------------
# 7. Make predictions on test data
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# Prediction Error Analysis
# --------------------------------------------------

error_analysis = pd.DataFrame({
    "Actual Performance": np.asarray(y_test).ravel(),
    "Predicted Performance": np.asarray(y_pred).ravel()
}, index=X_test.index)

error_analysis.index.name = "Original Dataset Index"

error_analysis["Residual"] = (
    error_analysis["Actual Performance"]
    - error_analysis["Predicted Performance"]
)

error_analysis["Absolute Residual"] = (
    error_analysis["Residual"].abs()
)

# Identify the five observations with the largest errors
worst_predictions = error_analysis.nlargest(
    5, "Absolute Residual"
)

worst_predictions.to_csv(
    OUTPUT_DIR / "worst_predictions.csv"
)

print("\nTOP 5 LARGEST PREDICTION ERRORS")
print(worst_predictions.round(2))



# --------------------------------------------------
# 8. Calculate evaluation metrics
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


# --------------------------------------------------
# 9. Display evaluation results
# --------------------------------------------------

print("\nMODEL EVALUATION RESULTS")

print("----------------------------")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)


# --------------------------------------------------
# 10. Save evaluation metrics
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
# 11. Actual vs Predicted graph
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
# 11. Residual Analysis
# --------------------------------------------------

# Calculate residuals (Actual Values - Predicted Values)
actual_values = np.asarray(y_test).ravel()
predicted_values = np.asarray(y_pred).ravel()
residuals = actual_values - predicted_values

# Residuals vs Predicted Values plot
plt.figure(figsize=(8, 5))

plt.scatter(predicted_values, residuals, alpha=0.7)
plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted Performance")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.grid(True, alpha=0.3)

plt.savefig(
    OUTPUT_DIR / "residuals_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Residual Distribution plot
plt.figure(figsize=(8, 5))

plt.hist(residuals, bins=15, edgecolor="black")
plt.axvline(x=0, linestyle="--")

plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")
plt.grid(True, alpha=0.3)

plt.savefig(
    OUTPUT_DIR / "residual_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Save residual statistics
residual_statistics = pd.DataFrame({
    "Metric": [
        "Mean Residual",
        "Standard Deviation",
        "Median Absolute Residual",
        "Maximum Absolute Residual"
    ],
    "Value": [
        np.mean(residuals),
        np.std(residuals),
        np.median(np.abs(residuals)),
        np.max(np.abs(residuals))
    ]
})

residual_statistics.to_csv(
    OUTPUT_DIR / "residual_statistics.csv",
    index=False
)

print("\nResidual analysis completed successfully.")



# --------------------------------------------------
# 12. Completion message
# --------------------------------------------------

print("\nEvaluation Completed Successfully!")

print("Evaluation metrics, cross-validation results, and residual analysis outputs saved successfully.")
