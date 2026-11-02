from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# =========================================================
# PATHS
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

INPUT_FILE = (
    PROCESSED_DIR / "groundwater_with_targets.csv"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# LOAD DATA
# =========================================================
print("\n" + "=" * 60)
print("AQUAINFORMER - MODEL TRAINING")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)


# =========================================================
# FEATURES
# =========================================================
# These are the actual water-quality measurements used
# by the baseline model.
FEATURES = [
    "ph",
    "ec_s_cm_at_25_c",
    "tds_mg_l",
    "total_alkalinity_mg_l",
    "cl_mg_l",
    "no3_mg_l",
    "so4_mg_l",
    "f_mg_l",
    "total_hardness_mg_l",
    "ca_mg_l",
    "mg_mg_l",
    "na_mg_l",
    "k_mg_l",
    "fe_mg_l",
    "as_ppb",
]


available_features = [
    column
    for column in FEATURES
    if column in df.columns
]


print("\nFeatures used:")

for feature in available_features:
    print(" -", feature)


X = df[available_features].copy()

y_quality = df["water_quality_class"]

y_risk = df["contamination_risk"]


# =========================================================
# CONVERT FEATURES TO NUMERIC
# =========================================================
for column in X.columns:
    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================
# Use the same split for both targets so their evaluation
# is directly comparable.

(
    X_train,
    X_test,
    y_quality_train,
    y_quality_test,
    y_risk_train,
    y_risk_test
) = train_test_split(
    X,
    y_quality,
    y_risk,
    test_size=0.20,
    random_state=42,
    stratify=y_quality
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# RANDOM FOREST
# =========================================================
def create_model():

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1
                )
            )
        ]
    )


# =========================================================
# WATER QUALITY MODEL
# =========================================================
print("\nTraining Water Quality model...")

quality_model = create_model()

quality_model.fit(
    X_train,
    y_quality_train
)


quality_model_path = (
    MODEL_DIR / "water_quality_model.joblib"
)

joblib.dump(
    quality_model,
    quality_model_path
)

print(
    "Saved:",
    quality_model_path
)


# =========================================================
# CONTAMINATION RISK MODEL
# =========================================================
print("\nTraining Contamination Risk model...")

risk_model = create_model()

risk_model.fit(
    X_train,
    y_risk_train
)


risk_model_path = (
    MODEL_DIR / "contamination_risk_model.joblib"
)

joblib.dump(
    risk_model,
    risk_model_path
)

print(
    "Saved:",
    risk_model_path
)


# =========================================================
# SAVE TEST DATA
# =========================================================
test_output = X_test.copy()

test_output["actual_water_quality"] = (
    y_quality_test.values
)

test_output["actual_contamination_risk"] = (
    y_risk_test.values
)

test_output_file = (
    PROCESSED_DIR / "test_set.csv"
)

test_output.to_csv(
    test_output_file,
    index=False
)


print(
    "\nSaved test set:",
    test_output_file
)


# =========================================================
# SUMMARY
# =========================================================
print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)

print("\nModels:")
print("1. water_quality_model.joblib")
print("2. contamination_risk_model.joblib")

print("\nNext step:")
print("Run 10_evaluate_model.py")