
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from src.preprocessing import build_preprocessing_pipeline


def get_models():
    """
    Define all five classification models.

    Returns:
        Dictionary containing model names and pipelines.
    """

    models = {
        "Logistic Regression": build_preprocessing_pipeline(
            LogisticRegression(max_iter=2000, random_state=42),
            scale=True
        ),

        "Decision Tree": build_preprocessing_pipeline(
            DecisionTreeClassifier(random_state=42),
            scale=False
        ),

        "K-Nearest Neighbors": build_preprocessing_pipeline(
            KNeighborsClassifier(n_neighbors=5),
            scale=True
        ),

        "Gaussian Naive Bayes": build_preprocessing_pipeline(
            GaussianNB(),
            scale=True
        ),

        "Support Vector Machine": build_preprocessing_pipeline(
            SVC(kernel="rbf", probability=True, random_state=42),
            scale=True
        )
    }

    return models


def train_models(X_train, y_train):
    """
    Train all five classifiers using the same training dataset.

    Args:
        X_train: Training features.
        y_train: Training target.

    Returns:
        Dictionary of fitted model pipelines.
    """

    models = get_models()
    trained_models = {}

    for model_name, model_pipeline in models.items():

        print(f"\nTraining {model_name}...")

        model_pipeline.fit(X_train, y_train)

        trained_models[model_name] = model_pipeline

        print(f"{model_name} trained successfully!")

    return trained_models
