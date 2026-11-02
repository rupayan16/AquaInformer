import pandas as pd
from pathlib import Path


# ============================================================
# AquaInformer - Step 1: Inspect Raw Datasets
# ============================================================

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw data directory
RAW_DATA = PROJECT_ROOT / "data" / "raw"


# ------------------------------------------------------------
# Dataset file paths
# ------------------------------------------------------------

datasets = {
    "Groundwater 2024": RAW_DATA / "west_bengal_gwq_2024.csv",

    "Surface Physical 1961-2020":
        RAW_DATA / "swq_manual_physical_parameters_cpcb_wb_1961_2020.csv",

    "Surface Physical 2021-2025":
        RAW_DATA / "swq_physical_parameter_manual_cpcb_wb_2021_2025.csv",

    "Surface Biological 1961-2020":
        RAW_DATA / "swq_manual_biological_parameters_cpcb_wb_1961_2020.csv",

    "Surface Biological 2021-2025":
        RAW_DATA / "swq_biological_parameter_manual_cpcb_wb_2021_2025.csv",
}


# ------------------------------------------------------------
# Inspect each dataset
# ------------------------------------------------------------

print("=" * 80)
print("AQUAINFORMER - RAW DATASET INSPECTION")
print("=" * 80)

for name, file_path in datasets.items():

    print("\n" + "=" * 80)
    print(f"DATASET: {name}")
    print("=" * 80)

    print(f"File: {file_path}")

    # Check whether file exists
    if not file_path.exists():
        print("STATUS: FILE NOT FOUND")
        continue

    print("STATUS: FILE FOUND")

    # Load dataset
    df = pd.read_csv(file_path, low_memory=False)

    # Basic information
    print("\n--- BASIC INFORMATION ---")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    # Column names
    print("\n--- COLUMN NAMES ---")
    for i, column in enumerate(df.columns, start=1):
        print(f"{i}. {column}")

    # Data types
    print("\n--- DATA TYPES ---")
    print(df.dtypes)

    # First 5 rows
    print("\n--- FIRST 5 ROWS ---")
    print(df.head())

    # Missing values
    print("\n--- MISSING VALUES ---")
    missing = df.isnull().sum()

    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) == 0:
        print("No missing values found.")
    else:
        print(missing)

    # Numeric summary
    print("\n--- NUMERIC SUMMARY ---")
    print(df.describe().T)

    print("\n")


print("=" * 80)
print("DATASET INSPECTION COMPLETED")
print("=" * 80)