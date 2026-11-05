
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model using six required metrics.

    Returns metrics, confusion matrix and classification report.
    """

    # Generate predictions
    y_pred = model.predict(X_test)

    # Get probabilities for malignant class (1)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate six required metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    # Detailed classification report
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Malignant"],
        output_dict=True,
        zero_division=0
    )

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report
    }


def evaluate_all_models(trained_models, X_test, y_test):
    """
    Evaluate all trained models.

    Returns:
        comparison_df: Six-metric comparison.
        detailed_results: Detailed results for each classifier.
    """

    comparison_results = {}
    detailed_results = {}

    for model_name, model in trained_models.items():

        print(f"\nEvaluating {model_name}...")

        result = evaluate_model(model, X_test, y_test)

        comparison_results[model_name] = result["metrics"]

        detailed_results[model_name] = {
            "confusion_matrix": result["confusion_matrix"],
            "classification_report": result["classification_report"]
        }

    comparison_df = pd.DataFrame(
        comparison_results
    ).T

    return comparison_df, detailed_results
