
from pathlib import Path

import joblib

from src.preprocessing import prepare_data
from src.model_training import train_models
from src.evaluation import evaluate_all_models


# Project directories
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
RESULTS_PATH = PROJECT_DIR / "model_comparison.csv"


def main():

    # Create model directory if it doesn't exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Prepare data
    print("\n========== DATA PREPARATION ==========")

    X_train, X_test, y_train, y_test = prepare_data()

    # Step 2: Train all five models
    print("\n========== MODEL TRAINING ==========")

    trained_models = train_models(X_train, y_train)

    # Step 3: Evaluate all models
    print("\n========== MODEL EVALUATION ==========")

    comparison_df = evaluate_all_models(
        trained_models,
        X_test,
        y_test
    )

    # Display comparison
    print("\n========== MODEL COMPARISON ==========")

    print(comparison_df.round(4).to_string())

    # Save comparison results
    comparison_df.to_csv(RESULTS_PATH)

    # Step 4: Save trained models
    print("\n========== SAVING MODELS ==========")

    model_files = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "K-Nearest Neighbors": "knn.joblib",
        "Gaussian Naive Bayes": "naive_bayes.joblib",
        "Support Vector Machine": "svm.joblib"
    }

    for model_name, model in trained_models.items():

        file_path = MODEL_DIR / model_files[model_name]

        joblib.dump(model, file_path)

        print(f"{model_name} saved to {file_path}")

    print("\n========== EXECUTION COMPLETED ==========")


if __name__ == "__main__":
    main()
