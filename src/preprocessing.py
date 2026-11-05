
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Dataset paths
DATASET_PATH = PROJECT_DIR / "dataset" / "breast_cancer.csv"
TEST_DATA_PATH = PROJECT_DIR / "test_data.csv"

# Configuration
TARGET_COLUMN = "diagnosis"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Target encoding
TARGET_MAPPING = {
    "B": 0,
    "M": 1
}


def load_dataset():
    """
    Load the prepared Breast Cancer dataset.
    """

    df = pd.read_csv(DATASET_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    return df


def prepare_data():
    """
    Encode target, split the dataset and save test data.

    Returns:
        X_train, X_test, y_train, y_test
    """

    # Load dataset
    df = load_dataset()

    # Validate target values
    if not set(df[TARGET_COLUMN].unique()).issubset(
        TARGET_MAPPING.keys()
    ):
        raise ValueError("Unexpected target values found.")

    # Encode target
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(
        TARGET_MAPPING
    )

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Create test dataset with features and actual labels
    test_data = X_test.copy()
    test_data[TARGET_COLUMN] = y_test

    # Save test dataset
    test_data.to_csv(TEST_DATA_PATH, index=False)

    # Display split information
    print("Data preprocessing completed successfully!")

    print("\nOriginal Dataset Shape:", df.shape)

    print("\nTraining Features Shape:", X_train.shape)
    print("Testing Features Shape:", X_test.shape)

    print("\nTraining Target Distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting Target Distribution:")
    print(y_test.value_counts().sort_index())

    print("\nSaved Test Dataset:", TEST_DATA_PATH)
    print("Test Dataset Shape:", test_data.shape)

    return X_train, X_test, y_train, y_test


def build_preprocessing_pipeline(classifier, scale=True):
    """
    Create a preprocessing and classification pipeline.

    Args:
        classifier: Scikit-learn classifier.
        scale: Whether StandardScaler should be applied.

    Returns:
        Scikit-learn Pipeline
    """

    steps = []

    if scale:
        steps.append(
            ("scaler", StandardScaler())
        )

    steps.append(
        ("classifier", classifier)
    )

    return Pipeline(steps)

if __name__ == "__main__":
    prepare_data()


