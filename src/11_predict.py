from pathlib import Path

import joblib
import pandas as pd


# =========================================================
# PATHS
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"


QUALITY_MODEL_FILE = (
    MODEL_DIR / "water_quality_model.joblib"
)

RISK_MODEL_FILE = (
    MODEL_DIR / "contamination_risk_model.joblib"
)


# =========================================================
# LOAD MODELS
# =========================================================
quality_model = joblib.load(
    QUALITY_MODEL_FILE
)

risk_model = joblib.load(
    RISK_MODEL_FILE
)


# =========================================================
# SAMPLE WATER-QUALITY INPUT
# =========================================================
# Replace these values later with actual sensor/API values.
#
# Units:
# pH                       -> pH
# EC                       -> µS/cm
# TDS                      -> mg/L
# Total Alkalinity         -> mg/L
# Chloride                 -> mg/L
# Nitrate                  -> mg/L
# Sulfate                  -> mg/L
# Fluoride                 -> mg/L
# Total Hardness           -> mg/L
# Calcium                  -> mg/L
# Magnesium                -> mg/L
# Sodium                   -> mg/L
# Potassium                -> mg/L
# Iron                     -> mg/L
# Arsenic                  -> ppb
# =========================================================

sample = pd.DataFrame(
    [{
        "ph": 7.2,
        "ec_s_cm_at_25_c": 800,
        "tds_mg_l": 450,
        "total_alkalinity_mg_l": 200,
        "cl_mg_l": 150,
        "no3_mg_l": 20,
        "so4_mg_l": 100,
        "f_mg_l": 0.6,
        "total_hardness_mg_l": 180,
        "ca_mg_l": 50,
        "mg_mg_l": 20,
        "na_mg_l": 80,
        "k_mg_l": 3,
        "fe_mg_l": 0.5,
        "as_ppb": 5,
    }]
)


# =========================================================
# PREDICTION
# =========================================================
quality_prediction = quality_model.predict(
    sample
)[0]

risk_prediction = risk_model.predict(
    sample
)[0]


# =========================================================
# PROBABILITIES
# =========================================================
quality_probabilities = (
    quality_model.predict_proba(sample)[0]
)

risk_probabilities = (
    risk_model.predict_proba(sample)[0]
)


# =========================================================
# DISPLAY
# =========================================================
print("\n" + "=" * 60)
print("AQUAINFORMER - WATER QUALITY PREDICTION")
print("=" * 60)

print("\nInput Water Quality:")
print(sample.to_string(index=False))


print("\n" + "-" * 60)

print(
    "Water Quality Classification:",
    quality_prediction
)

print(
    "Contamination Risk:",
    risk_prediction
)


print("\n" + "-" * 60)

print("Water Quality Probabilities:")

for label, probability in zip(
    quality_model.classes_,
    quality_probabilities
):
    print(
        f"{label}: {probability:.3f}"
    )


print("\nContamination Risk Probabilities:")

for label, probability in zip(
    risk_model.classes_,
    risk_probabilities
):
    print(
        f"{label}: {probability:.3f}"
    )


print("\n" + "=" * 60)
print("PREDICTION COMPLETED")
print("=" * 60)
