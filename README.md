# Breast Cancer Classification – Machine Learning Assignment 2

**Programme:** M.Tech Artificial Intelligence and Machine Learning  
**Course:** Machine Learning  
**Assignment:** Assignment 2 – Classification Models and Evaluation

---

## a. Problem Statement

Breast cancer is one of the most commonly diagnosed cancers worldwide. Early identification and classification of breast tumours can support timely medical assessment and treatment planning.

The objective of this project is to develop and evaluate multiple machine learning classification models to distinguish between benign and malignant breast tumours using the Breast Cancer Wisconsin Diagnostic dataset.

Five classification algorithms are implemented and evaluated on the same dataset. Their performance is compared using six evaluation metrics to identify the best-performing classifier.

An interactive Streamlit application is also developed to allow users to upload test data, select a trained classification model and view its evaluation results.

**Objective:**
- Implement five machine learning classification algorithms.
- Apply suitable data preprocessing techniques.
- Evaluate each model using six performance metrics.
- Compare the performance of all five classifiers.
- Deploy an interactive Streamlit application using the saved trained models.

---

## b. Dataset Description

**Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset

**Source:** UCI Machine Learning Repository

**Dataset Link:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

### Dataset Overview

The Breast Cancer Wisconsin Diagnostic dataset contains measurements computed from digitized images of fine needle aspirates (FNA) of breast masses.

The features describe characteristics of cell nuclei present in the images.

| Dataset Attribute | Description |
|---|---|
| Total Instances | 569 |
| Number of Input Features | 30 |
| Target Column | diagnosis |
| Classification Type | Binary Classification |
| Class 0 | Benign |
| Class 1 | Malignant |
| Missing Values | 0 |

### Feature Description

The dataset contains 30 numerical input features derived from ten characteristics of cell nuclei:

1. Radius
2. Texture
3. Perimeter
4. Area
5. Smoothness
6. Compactness
7. Concavity
8. Concave Points
9. Symmetry
10. Fractal Dimension

These characteristics are represented through three measurements:

- Mean values
- Standard Error values
- Worst values

### Target Distribution

| Diagnosis | Original Dataset Count |
|---|---:|
| Benign (B) | 357 |
| Malignant (M) | 212 |
| **Total** | **569** |

### Data Preprocessing

The following preprocessing operations were performed:

- Removed the unnecessary ID column.
- Converted the dataset into a structured CSV format.
- Encoded the target variable into numerical values:
  - Benign = 0
  - Malignant = 1
- Separated input features and the target variable.
- Split the dataset into training and testing sets using an 80:20 ratio.
- Applied StandardScaler where required through Scikit-learn pipelines.

### Training and Testing Dataset

| Dataset | Number of Records |
|---|---:|
| Training Dataset | 455 |
| Testing Dataset | 114 |
| Total | 569 |

The test dataset is saved as `test_data.csv` and is used for evaluating the trained classification models.

---

## c. GitHub Repository Link

**GitHub Repository:** [ADD YOUR GITHUB REPOSITORY LINK HERE]

The repository contains the complete source code, preprocessing scripts, trained classification models, test dataset, evaluation results, requirements file and Streamlit application.

### Project Structure

```text
ML_Assignment_2/
│
├── app.py
├── train_models.py
├── evaluate_saved_models.py
├── requirements.txt
├── README.md
├── test_data.csv
│
├── dataset/
│   ├── breast_cancer.csv
│   ├── wdbc.data
│   └── wdbc.names
│
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
│
├── evaluation_results/
│   └── Evaluation reports and confusion matrices
│
├── notebooks/
│   └── eda.ipynb
│
└── src/
    ├── __init__.py
    ├── prepare_dataset.py
    ├── preprocessing.py
    ├── model_training.py
    └── evaluation.py
```

---

## d. Models Used and Evaluation Metrics

Five machine learning classification algorithms were implemented using the same Breast Cancer Wisconsin Diagnostic dataset.

### 1. Logistic Regression

Logistic Regression is a supervised classification algorithm that estimates the probability of a sample belonging to a particular class using the logistic function.

### 2. Decision Tree Classifier

Decision Tree is a supervised learning algorithm that classifies observations by making decisions based on feature values and splitting the dataset into branches.

### 3. K-Nearest Neighbors (KNN)

KNN is a distance-based classification algorithm that predicts the class of a sample based on the majority class among its nearest neighbours.

### 4. Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem. It assumes that continuous features follow a Gaussian distribution within each class.

### 5. Support Vector Machine (SVM)

Support Vector Machine is a supervised learning algorithm that identifies an optimal decision boundary to separate different classes. An RBF kernel was used for classification.

### Evaluation Metrics

All five classifiers were evaluated using the following six metrics:

- **Accuracy:** Measures the proportion of correctly classified instances.
- **AUC Score:** Measures the model's ability to distinguish between the two classes across different classification thresholds.
- **Precision:** Measures the proportion of correctly predicted positive instances among all predicted positive instances.
- **Recall:** Measures the proportion of actual positive instances correctly identified by the model.
- **F1 Score:** Represents the harmonic mean of precision and recall.
- **Matthews Correlation Coefficient (MCC):** Measures the quality of binary classifications using all four confusion matrix categories.

### Model Comparison Table

The following table presents the baseline evaluation results obtained from the held-out test dataset containing 114 records.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Achieved an accuracy of 96.49% and the highest AUC score of 0.9960. It demonstrated strong classification performance and a good balance between precision and recall. |
| Decision Tree | Achieved an accuracy of 92.98%. It produced comparatively lower AUC and MCC scores than the other classifiers, indicating weaker overall baseline performance on this test dataset. |
| KNN | Achieved an accuracy of 95.61% with a precision of 97.44%. It demonstrated good classification performance, although its recall was slightly lower than Logistic Regression and SVM. |
| Gaussian Naive Bayes | Achieved an accuracy of 92.11%. It obtained an AUC score of 0.9891 but had the lowest recall of 85.71%, indicating that it missed more malignant cases than the other classifiers. |
| Support Vector Machine | Achieved the highest accuracy of 97.37%, precision of 100%, F1 score of 0.9630 and MCC of 0.9442. It demonstrated the strongest overall baseline performance among the five classifiers. |

### Overall Winner for the Dataset

**Support Vector Machine (SVM)**

Based on the baseline evaluation results, Support Vector Machine was identified as the best-performing classifier overall.

It achieved:

- Accuracy: 97.37%
- AUC: 0.9947
- Precision: 100%
- Recall: 92.86%
- F1 Score: 0.9630
- MCC: 0.9442

SVM achieved the highest accuracy, precision, F1 score and MCC among the five evaluated classifiers.

Logistic Regression achieved the highest AUC score of 0.9960, demonstrating that it also performed exceptionally well.

The comparison is based on the selected test dataset and does not establish clinical suitability or guarantee performance on external data.

---

## e. Streamlit Application

An interactive web application was developed using Streamlit to demonstrate the trained classification models.

### Application Features

1. **Dataset Upload:** Allows users to upload the prepared `test_data.csv` file.
2. **Dataset Validation:** Checks whether the uploaded CSV contains the required 30 numerical features and the diagnosis target column.
3. **Model Selection:** Provides a dropdown to select one of the five trained classification models.
4. **Evaluation Metrics:** Displays Accuracy, AUC, Precision, Recall, F1 Score and MCC.
5. **Confusion Matrix:** Displays the confusion matrix for the selected classifier.
6. **Classification Report:** Displays class-wise precision, recall and F1 scores.
7. **Model Comparison:** Displays evaluation results for all five classifiers on the uploaded test dataset.
8. **Download Results:** Allows users to download the model comparison results as a CSV file.

### How to Use the Application

1. Open the deployed Streamlit application.
2. Download `test_data.csv` from the root directory of the GitHub repository.
3. Upload the downloaded CSV using the dataset uploader.
4. Select a classification model from the dropdown.
5. Click the **Evaluate Model** button.
6. View the evaluation metrics, confusion matrix and classification report.
7. Compare the performance of all five classification models.

The application loads the previously trained and saved model files. It does not retrain any classifier using the uploaded test dataset.

**Streamlit Application Link:** [ADD YOUR DEPLOYED STREAMLIT APP LINK HERE]

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Jupyter Notebook

---

## Conclusion

This project demonstrates an end-to-end machine learning classification workflow using the Breast Cancer Wisconsin Diagnostic dataset.

Five classification algorithms were implemented and evaluated using six performance metrics. The results showed that Support Vector Machine achieved the strongest overall baseline performance, while Logistic Regression obtained the highest AUC score.

The trained models were saved and integrated into an interactive Streamlit application, allowing users to upload compatible test data and compare classifier performance without retraining the models.

