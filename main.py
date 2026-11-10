
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

scripts = [
    NOTEBOOKS_DIR / "data_preparation.py",
    NOTEBOOKS_DIR / "data_visualisation.py",
    NOTEBOOKS_DIR / "model_preparation.py",
    NOTEBOOKS_DIR / "model_evaluation.py"
]

expected_outputs = [
    "cleaned_dataset.csv",
    "correlation_heatmap.png",
    "feature_relationships.png",
    "trained_model.pkl",
    "evaluation_metrics.csv",
    "actual_vs_predicted.png"
]


def main():

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

    print("\n========== PROJECT OUTPUTS ==========\n")

    for filename in expected_outputs:
        file_path = OUTPUT_DIR / filename

        if file_path.exists():
            print(f"[OK] {filename}")
        else:
            print(f"[MISSING] {filename}")

    metrics_path = OUTPUT_DIR / "evaluation_metrics.csv"

    if metrics_path.exists():
        print("\n========== EVALUATION RESULTS ==========\n")
        print(metrics_path.read_text(encoding="utf-8"))

    print("\n========== EXECUTION COMPLETED ==========")


if __name__ == "__main__":
    main()
