
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


steps = [
    (
        "MEMBER 1 - DATA PREPROCESSING",
        NOTEBOOKS_DIR / "data_preparation.py"
    ),
    (
        "MEMBER 2 - DATA ANALYSIS AND VISUALISATION",
        NOTEBOOKS_DIR / "data_visualisation.py"
    ),
    (
        "MEMBER 3 - MODEL TRAINING",
        NOTEBOOKS_DIR / "model_preparation.py"
    ),
    (
        "MEMBER 4 - MODEL EVALUATION",
        NOTEBOOKS_DIR / "model_evaluation.py"
    )
]

def main():

    print("\n==========================================")
    print(" CPU PERFORMANCE PREDICTION PROJECT")
    print("==========================================")

    for step_name, script_path in steps:

        print("\n\n------------------------------------------")
        print("Running:", step_name)
        print("------------------------------------------", flush=True)

        if not script_path.is_file():
            print("\nERROR: File not found:", script_path)
            sys.exit(1)

        try:
            subprocess.run(
                [sys.executable, str(script_path)],
                cwd=PROJECT_DIR,
                check=True
            )

            print("\nCompleted:", step_name)

        except subprocess.CalledProcessError as error:

            print("\nERROR: Project execution stopped.")
            print("Failed module:", step_name)
            print("Exit code:", error.returncode)

            sys.exit(error.returncode)


    print("\n\n==========================================")
    print(" FINAL PROJECT RESULTS")
    print("==========================================")

    metrics_path = OUTPUT_DIR / "evaluation_metrics.csv"

    if metrics_path.exists():

        print("\nModel Evaluation Metrics:\n")

        print(metrics_path.read_text(encoding="utf-8"))

    else:
        print("\nEvaluation metrics file was not generated.")


    print("\nGenerated Project Outputs:")

    expected_outputs = [
        "cleaned_dataset.csv",
        "correlation_heatmap.png",
        "feature_relationships.png",
        "trained_model.pkl",
        "evaluation_metrics.csv",
        "actual_vs_predicted.png"
    ]

    for filename in expected_outputs:

        file_path = OUTPUT_DIR / filename

        if file_path.exists():
            print("[OK]", filename)
        else:
            print("[MISSING]", filename)


    print("\n==========================================")
    print(" PROJECT EXECUTION COMPLETED SUCCESSFULLY")
    print("==========================================")

    print("\nAll outputs are available in the outputs folder."

          
if __name__ == "__main__":
    main()
