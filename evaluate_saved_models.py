
from pathlib import Path

import joblib
import pandas as pd

from src.evaluation import evaluate_all_models


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_DIR / "model"
TEST_DATA_PATH = PROJECT_DIR / "test_data.csv"
REPORT_DIR = PROJECT_DIR / "evaluation_results"

TARGET_COLUMN = "diagnosis"

# Model file mapping
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}


def main():

    # Create evaluation results directory
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Load test dataset
    test_data = pd.read_csv(TEST_DATA_PATH)

    X_test = test_data.drop(columns=[TARGET_COLUMN])
    y_test = test_data[TARGET_COLUMN]

    trained_models = {}

    # Load existing saved models
    print("\n========== LOADING SAVED MODELS ==========")

    for model_name, filename in MODEL_FILES.items():

        model_path = MODEL_DIR / filename

        trained_models[model_name] = joblib.load(model_path)

        print(f"{model_name} loaded successfully!")

    # Evaluate without retraining
    print("\n========== EVALUATING SAVED MODELS ==========")

    comparison_df, detailed_results = evaluate_all_models(
        trained_models,
        X_test,
        y_test
    )

    # Save six-metric comparison
    comparison_path = (
        PROJECT_DIR / "model_comparison.csv"
    )

    comparison_df.to_csv(comparison_path)

    print("\n========== MODEL COMPARISON ==========")
    print(comparison_df.round(4).to_string())

    # Save detailed results for each model
    for model_name, results in detailed_results.items():

        filename_prefix = (
            model_name.lower().replace(" ", "_").replace("-", "_")
        )

        # Save confusion matrix
        cm_df = pd.DataFrame(
            results["confusion_matrix"],
            index=["Actual Benign", "Actual Malignant"],
            columns=["Predicted Benign", "Predicted Malignant"]
        )

        cm_df.to_csv(
            REPORT_DIR / f"{filename_prefix}_confusion_matrix.csv"
        )

        # Save classification report
        report_df = pd.DataFrame(
            results["classification_report"]
        ).transpose()

        report_df.to_csv(
            REPORT_DIR / f"{filename_prefix}_classification_report.csv"
        )

    print("\nAll evaluation reports saved successfully!")
    print("Results directory:", REPORT_DIR)


if __name__ == "__main__":
    main()
