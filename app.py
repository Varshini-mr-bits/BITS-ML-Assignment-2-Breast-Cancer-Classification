
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

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


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_DIR / "model"
DATASET_PATH = PROJECT_DIR / "dataset" / "breast_cancer.csv"

TARGET_COLUMN = "diagnosis"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}


# ==================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🧬",
    layout="wide"
)


# ==================================================
# 3. LOAD EXPECTED FEATURES
# ==================================================

@st.cache_data
def get_expected_features():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "The prepared dataset is missing from the dataset folder."
        )

    df = pd.read_csv(DATASET_PATH, nrows=1)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            "Diagnosis column is missing from the reference dataset."
        )

    return [
        column
        for column in df.columns
        if column != TARGET_COLUMN
    ]


# ==================================================
# 4. LOAD SAVED MODELS
# ==================================================

@st.cache_resource
def load_models():

    models = {}

    for model_name, filename in MODEL_FILES.items():

        model_path = MODEL_DIR / filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file missing: {filename}"
            )

        models[model_name] = joblib.load(model_path)

    return models


# ==================================================
# 5. VALIDATE UPLOADED DATASET
# ==================================================

def validate_dataset(df, expected_features):

    if df.empty:
        return False, "The uploaded CSV is empty."

    if df.columns.duplicated().any():
        return False, "Duplicate column names detected."

    expected_columns = set(expected_features + [TARGET_COLUMN])
    uploaded_columns = set(df.columns)

    missing_columns = expected_columns - uploaded_columns
    extra_columns = uploaded_columns - expected_columns

    if missing_columns:
        return False, (
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if extra_columns:
        return False, (
            f"Unexpected columns found: {sorted(extra_columns)}"
        )

    # Validate numerical features
    try:
        for column in expected_features:
            df[column] = pd.to_numeric(
                df[column],
                errors="raise"
            )
    except (ValueError, TypeError):
        return False, "All 30 input features must be numerical."

    # Check missing and infinite feature values
    feature_values = df[expected_features].to_numpy(dtype=float)

    if not np.isfinite(feature_values).all():
        return False, (
            "Missing, infinite or invalid values found in input features."
        )

    # Validate target column
    try:
        df[TARGET_COLUMN] = pd.to_numeric(
            df[TARGET_COLUMN],
            errors="raise"
        )
    except (ValueError, TypeError):
        return False, "Diagnosis must contain numerical values 0 or 1."

    if df[TARGET_COLUMN].isnull().any():
        return False, "Missing diagnosis values found."

    if not df[TARGET_COLUMN].isin([0, 1]).all():
        return False, (
            "Diagnosis must contain only 0 (Benign) or 1 (Malignant)."
        )

    if df[TARGET_COLUMN].nunique() != 2:
        return False, (
            "Evaluation requires both Benign and Malignant classes."
        )

    return True, "Dataset validation successful!"


# ==================================================
# 6. MODEL EVALUATION
# ==================================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y, predictions),

        "AUC": roc_auc_score(y, probabilities),

        "Precision": precision_score(
            y, predictions, zero_division=0
        ),

        "Recall": recall_score(
            y, predictions, zero_division=0
        ),

        "F1 Score": f1_score(
            y, predictions, zero_division=0
        ),

        "MCC": matthews_corrcoef(y, predictions)
    }

    return metrics, predictions


# ==================================================
# 7. APPLICATION HEADER
# ==================================================

st.title("🧬 Breast Cancer Classification")

st.caption(
    "Machine Learning Model Evaluation Dashboard"
)

st.info(
    """
    This application evaluates five pre-trained machine learning
    classifiers using the Breast Cancer Wisconsin Diagnostic dataset.

    The application loads saved models and does not retrain them.
    """
)


# ==================================================
# 8. INITIALIZE MODELS
# ==================================================

try:

    models = load_models()
    expected_features = get_expected_features()

except Exception as error:

    st.error(f"Application initialization failed: {error}")
    st.stop()


# ==================================================
# 9. DATASET UPLOAD INSTRUCTIONS
# ==================================================

st.header("1. Upload Test Dataset")

st.markdown(
    """
    To evaluate the trained classification models, please follow
    these steps:

    1. Open the project's GitHub repository.
    2. Download **`test_data.csv`** from the root directory.
    3. Upload the downloaded file using the uploader below.
    4. Select a classification model and click **Evaluate Model**.

    **Important:** Only the Breast Cancer Wisconsin test dataset
    generated for this assignment is accepted.
    """
)

with st.expander("What should the uploaded CSV contain?"):

    st.markdown(
        """
        Your CSV must contain:

        - 30 numerical Breast Cancer input features.
        - One target column named `diagnosis`.
        - Target values: `0` (Benign) and `1` (Malignant).
        - No missing or infinite values.
        - No additional or missing columns.

        The supplied `test_data.csv` contains 114 records and
        31 columns.
        """
    )

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"],
    help="Download test_data.csv from the GitHub repository."
)


# ==================================================
# 10. PROCESS UPLOADED DATASET
# ==================================================

if uploaded_file is not None:

    try:

        uploaded_data = pd.read_csv(uploaded_file)

        # Validate uploaded data
        is_valid, message = validate_dataset(
            uploaded_data,
            expected_features
        )

        if not is_valid:

            st.error(message)
            st.stop()

        st.success(message)

        # Reorder columns to match training feature order
        uploaded_data = uploaded_data[
            expected_features + [TARGET_COLUMN]
        ]

        st.subheader("Uploaded Dataset Preview")

        st.dataframe(
            uploaded_data.head(),
            width="stretch",
            height=180
        )

        st.caption(
            f"Total Records: {len(uploaded_data)} | "
            f"Input Features: {len(expected_features)} | "
            f"Target Column: {TARGET_COLUMN}"
        )

        # Separate features and target
        X_test = uploaded_data[expected_features].copy()

        y_test = uploaded_data[TARGET_COLUMN].astype(int)

        # ==================================================
        # 11. MODEL SELECTION
        # ==================================================

        st.header("2. Select Classification Model")

        selected_model = st.selectbox(
            "Choose a classifier",
            list(models.keys())
        )

        model = models[selected_model]

        # ==================================================
        # 12. EVALUATE SELECTED MODEL
        # ==================================================

        if st.button(
            "Evaluate Model",
            type="primary"
        ):

            metrics, predictions = evaluate_model(
                model,
                X_test,
                y_test
            )

            st.header("3. Evaluation Results")

            # Display six evaluation metrics
            metric_columns = st.columns(3)

            for index, (metric, value) in enumerate(
                metrics.items()
            ):

                metric_columns[index % 3].metric(
                    metric,
                    f"{value:.4f}"
                )

            # ==================================================
            # 13. CONFUSION MATRIX
            # ==================================================

            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                y_test,
                predictions,
                labels=[0, 1]
            )

            fig, ax = plt.subplots(
                figsize=(4, 3),
                dpi=100
            )

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                square=True,
                linewidths=0.5,
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"],
                annot_kws={"size": 11},
                ax=ax
            )

            ax.set_xlabel(
                "Predicted Diagnosis",
                fontsize=9
            )

            ax.set_ylabel(
                "Actual Diagnosis",
                fontsize=9
            )

            ax.tick_params(
                axis="both",
                labelsize=9
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                width="content"
            )

            plt.close(fig)

            # ==================================================
            # 14. CLASSIFICATION REPORT
            # ==================================================

            st.subheader("Classification Report")

            report = classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["Benign", "Malignant"],
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(report).transpose()

            st.dataframe(
                report_df.round(4),
                width="stretch",
                height=210
            )

            # ==================================================
            # 15. COMPARE ALL FIVE CLASSIFIERS
            # ==================================================

            st.header("4. Compare All Five Classifiers")

            comparison = {}

            for name, saved_model in models.items():

                result, _ = evaluate_model(
                    saved_model,
                    X_test,
                    y_test
                )

                comparison[name] = result

            comparison_df = pd.DataFrame(comparison).T

            st.dataframe(
                comparison_df.style.format("{:.4f}"),
                width="stretch"
            )

            # ==================================================
            # 16. ACCURACY COMPARISON CHART
            # ==================================================

            st.subheader("Accuracy Comparison")

            accuracy_values = (
                comparison_df["Accuracy"].sort_values()
            )

            fig, ax = plt.subplots(
                figsize=(7, 3.5),
                dpi=100
            )

            accuracy_values.plot(
                kind="barh",
                ax=ax,
                width=0.6
            )

            ax.set_xlabel(
                "Accuracy",
                fontsize=9
            )

            ax.set_ylabel(
                "Classifier",
                fontsize=9
            )

            ax.set_xlim(0, 1.05)

            ax.tick_params(
                axis="both",
                labelsize=9
            )

            for container in ax.containers:

                ax.bar_label(
                    container,
                    fmt="%.3f",
                    padding=3,
                    fontsize=8
                )

            fig.tight_layout()

            st.pyplot(
                fig,
                width="content"
            )

            plt.close(fig)

            # ==================================================
            # 17. DOWNLOAD EVALUATION RESULTS
            # ==================================================

            st.subheader("Download Evaluation Results")

            csv_data = comparison_df.to_csv().encode("utf-8")

            st.download_button(
                label="Download Model Comparison CSV",
                data=csv_data,
                file_name="model_comparison.csv",
                mime="text/csv"
            )

    except (ValueError, TypeError, pd.errors.ParserError) as error:

        st.error(f"Unable to process CSV: {error}")

    except Exception as error:

        st.error(f"Evaluation failed: {error}")


else:

    st.warning(
        "Please download test_data.csv from the GitHub repository "
        "and upload it here to begin evaluation."
    )

    st.caption(
        "The application cannot evaluate arbitrary datasets. "
        "Only the expected Breast Cancer Wisconsin test dataset "
        "is supported."
    )
