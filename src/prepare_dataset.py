import pandas as pd
from pathlib import Path

# Define project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"

# Dataset feature names
features = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",

    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",

    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]

columns = ["id", "diagnosis"] + features

# Load original UCI dataset
data_path = DATASET_DIR / "wdbc.data"

df = pd.read_csv(
    data_path,
    header=None,
    names=columns
)

# Remove identifier column
df.drop(columns=["id"], inplace=True)

# Save cleaned dataset
output_path = DATASET_DIR / "breast_cancer.csv"

df.to_csv(output_path, index=False)

# Verify dataset
print("Dataset successfully converted!")

print("\nDataset Shape:", df.shape)

print("\nFirst five records:")
print(df.head())

print("\nTarget Distribution:")
print(df["diagnosis"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nNumber of Input Features:", len(df.columns) - 1)

# Validate assignment requirements
assert len(df) >= 500
assert len(df.columns) - 1 >= 12
assert df["diagnosis"].nunique() == 2

print("\nAll assignment dataset requirements satisfied!")