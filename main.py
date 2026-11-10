
import subprocess
import sys
from pathlib import Path



# Define project directories
PROJECT_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"
OUTPUT_DIR = PROJECT_DIR / "outputs"


# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# List all Python scripts in execution order
scripts = [
    NOTEBOOKS_DIR / "data_preparation.py",
    NOTEBOOKS_DIR / "data_visualisation.py",
    NOTEBOOKS_DIR / "model_preparation.py",
    NOTEBOOKS_DIR / "model_evaluation.py"
]


# Expected files generated after successful execution
expected_outputs = [
    "cleaned_dataset.csv",
    "correlation_heatmap.png",
    "feature_relationships.png",
    "trained_model.pkl",
    "evaluation_metrics.csv",
    "actual_vs_predicted.png"
]


def main():

    # Execute each project script sequentially
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

        
        # Stop execution if any script encounters an error
        if result.returncode != 0:
            print("\nERROR: Project execution failed.")

            if result.stdout:
                print(result.stdout)

            if result.stderr:
                print(result.stderr)

            sys.exit(result.returncode)

    
    # Display the status of all expected output files
    print("\n========== PROJECT OUTPUTS ==========\n")

    for filename in expected_outputs:
        file_path = OUTPUT_DIR / filename

        if file_path.exists():
            print(f"[OK] {filename}")
        else:
            print(f"[MISSING] {filename}")

     # Display evaluation metrics if the CSV file exists
    metrics_path = OUTPUT_DIR / "evaluation_metrics.csv"

    if metrics_path.exists():
        print("\n========== EVALUATION RESULTS ==========\n")
        print(metrics_path.read_text(encoding="utf-8"))

    print("\n========== EXECUTION COMPLETED ==========")


# Run the main function when this file is executed directly
if __name__ == "__main__":
    main()
