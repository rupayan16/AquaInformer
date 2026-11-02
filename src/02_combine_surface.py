import pandas as pd
from pathlib import Path


# ============================================================
# AquaInformer - Step 2: Combine Surface Physical Datasets
# ============================================================

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directories
RAW_DATA = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"

# Create processed directory if it does not exist
PROCESSED_DATA.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

file_1961_2020 = (
    RAW_DATA
    / "swq_manual_physical_parameters_cpcb_wb_1961_2020.csv"
)

file_2021_2025 = (
    RAW_DATA
    / "swq_physical_parameter_manual_cpcb_wb_2021_2025.csv"
)


# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

print("=" * 80)
print("AQUAINFORMER - COMBINING SURFACE PHYSICAL DATA")
print("=" * 80)

df_1961_2020 = pd.read_csv(
    file_1961_2020,
    low_memory=False
)

df_2021_2025 = pd.read_csv(
    file_2021_2025,
    low_memory=False
)


print("\nOriginal dataset sizes:")
print("1961-2020:", df_1961_2020.shape)
print("2021-2025:", df_2021_2025.shape)


# ------------------------------------------------------------
# Add source-period information
# ------------------------------------------------------------

df_1961_2020["Dataset_Period"] = "1961-2020"
df_2021_2025["Dataset_Period"] = "2021-2025"


# ------------------------------------------------------------
# Combine datasets
# ------------------------------------------------------------

combined = pd.concat(
    [df_1961_2020, df_2021_2025],
    ignore_index=True
)


print("\nCombined dataset size before duplicate removal:")
print(combined.shape)


# ------------------------------------------------------------
# Remove exact duplicate rows
# ------------------------------------------------------------

duplicate_count = combined.duplicated().sum()

print("\nExact duplicate rows found:", duplicate_count)

if duplicate_count > 0:
    combined = combined.drop_duplicates().reset_index(drop=True)


print("Combined dataset size after duplicate removal:")
print(combined.shape)


# ------------------------------------------------------------
# Check Dataset_Period
# ------------------------------------------------------------

print("\nDataset period distribution:")
print(combined["Dataset_Period"].value_counts())


# ------------------------------------------------------------
# Save combined dataset
# ------------------------------------------------------------

output_file = (
    PROCESSED_DATA
    / "west_bengal_surface_physical_combined.csv"
)

combined.to_csv(
    output_file,
    index=False
)


# ------------------------------------------------------------
# Final information
# ------------------------------------------------------------

print("\nOutput file created:")
print(output_file)

print("\nFinal dataset shape:")
print(combined.shape)

print("\nFinal columns:")
for i, column in enumerate(combined.columns, start=1):
    print(f"{i}. {column}")

print("\n" + "=" * 80)
print("SURFACE PHYSICAL DATA COMBINATION COMPLETED")
print("=" * 80)