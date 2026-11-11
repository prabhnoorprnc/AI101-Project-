
import subprocess
import sys
from pathlib import Path

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Scripts execution order
scripts = [
    NOTEBOOKS_DIR / "data_preparation.py",
    NOTEBOOKS_DIR / "data_visualisation.py",
    NOTEBOOKS_DIR / "model_preparation.py",
    NOTEBOOKS_DIR / "model_evaluation.py"
]

# Expected output files
expected_outputs = [
    Path("Data_Processing/cleaned_dataset.csv"),

    Path("Data_Analysis_Visualisation/correlation_heatmap.png"),
    Path("Data_Analysis_Visualisation/feature_relationships.png"),

    Path("Model_Training/linear_regression_model.pkl"),
    Path("Model_Training/evaluation_metrics.csv"),
    Path("Model_Training/cross_validation_results.csv"),
    Path("Model_Training/worst_predictions.csv"),
    Path("Model_Training/actual_vs_predicted.png"),
    Path("Model_Training/residuals_vs_predicted.png"),
    Path("Model_Training/residual_distribution.png"),
    Path("Model_Training/residual_statistics.csv")
]


def main():

    # Execute all scripts
    for script in scripts:

        if not script.exists():
            print(f"\nERROR: {script.name} not found.")
            sys.exit(1)

        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("\nERROR: Project execution failed.")

            if result.stdout:
                print(result.stdout)

            if result.stderr:
                print(result.stderr)

            sys.exit(result.returncode)

    # Display generated output files
    print("\n========== GENERATED OUTPUTS ==========\n")

    for filename in expected_outputs:

        file_path = OUTPUT_DIR / filename

        if file_path.exists():
            print(f"[OK] {filename}")
        else:
            print(f"[MISSING] {filename}")

    # Display important evaluation CSV files
    evaluation_files = [
        "evaluation_metrics.csv",
        "cross_validation_results.csv",
        "worst_predictions.csv",
        "residual_statistics.csv"
    ]

    for filename in evaluation_files:

        file_path = OUTPUT_DIR / "Model_Training" / filename

        if file_path.exists():
            print(f"\n========== {filename} ==========\n")
            print(file_path.read_text(encoding="utf-8"))

    print("\n========== PROJECT COMPLETED ==========")


if __name__ == "__main__":
    main()
