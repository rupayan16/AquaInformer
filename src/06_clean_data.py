import pandas as pd
from pathlib import Path


# ============================================================
# AquaInformer - Step 6: Data Cleaning
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"

PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper functions
# ============================================================

def clean_column_names(df):
    """
    Standardize column names without changing their meaning.
    """
    df.columns = (
        df.columns
        .str.replace("\n", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return df


def convert_numeric_columns(df, columns):
    """
    Convert selected columns to numeric.
    Non-numeric entries become NaN.
    """
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


def remove_empty_columns(df):
    """
    Remove columns containing no usable values at all.
    """
    before = df.shape[1]

    df = df.dropna(
        axis=1,
        how="all"
    )

    after = df.shape[1]

    print(
        f"Removed {before - after} completely empty columns."
    )

    return df


def save_clean_dataset(df, output_file):
    """
    Save a cleaned dataset.
    """
    df.to_csv(
        output_file,
        index=False
    )

    print("Saved:", output_file)
    print("Shape:", df.shape)


# ============================================================
# START
# ============================================================

print("=" * 80)
print("AQUAINFORMER - DATA CLEANING")
print("=" * 80)


# ============================================================
# PART 1 - GROUNDWATER
# ============================================================

print("\n" + "=" * 80)
print("1. GROUNDWATER CLEANING")
print("=" * 80)


groundwater_file = RAW_DATA / "west_bengal_gwq_2024.csv"

groundwater = pd.read_csv(
    groundwater_file,
    low_memory=False
)

print("Original shape:", groundwater.shape)


# ------------------------------------------------------------
# Standardize column names
# ------------------------------------------------------------

groundwater = clean_column_names(
    groundwater
)


# ------------------------------------------------------------
# Replace known missing-value markers
# ------------------------------------------------------------
#
# IMPORTANT:
# BDL = Below Detection Limit.
# We do NOT convert BDL to zero.
#
# "_" is treated as missing because the dataset uses it
# as a missing-value marker.
# ------------------------------------------------------------

groundwater = groundwater.replace(
    {
        "_": pd.NA,
        "BDL": pd.NA,
        "bdl": pd.NA,
        "NA": pd.NA,
        "N/A": pd.NA,
        "na": pd.NA,
        "": pd.NA,
        " ": pd.NA
    }
)


# ------------------------------------------------------------
# Groundwater numeric columns
# ------------------------------------------------------------

groundwater_numeric_columns = [
    "pH",
    "EC (µ S-cm at 25 ° C)",
    "TDS (mg/L)",
    "CO3 (mg/L)",
    "HCO3 (mg/L)",
    "Total Alkalinity (mg/L)",
    "Cl (mg/L)",
    "NO3 (mg/L)",
    "SO4 (mg/L)",
    "PO4 (mg/L)",
    "SiO2 (mg/L)",
    "F (mg/L)",
    "Total Hardness (mg/L)",
    "Ca (mg/L)",
    "Mg (mg/L)",
    "Na (mg/L)",
    "K (mg/L)",
    "Fe (mg/L)",
    "As (ppb)"
]

groundwater = convert_numeric_columns(
    groundwater,
    groundwater_numeric_columns
)


# ------------------------------------------------------------
# Remove completely empty columns
# ------------------------------------------------------------

groundwater = remove_empty_columns(
    groundwater
)


# ------------------------------------------------------------
# Remove exact duplicate rows
# ------------------------------------------------------------

duplicate_count = groundwater.duplicated().sum()

print(
    "Duplicate groundwater rows:",
    duplicate_count
)

if duplicate_count > 0:
    groundwater = groundwater.drop_duplicates(
        ignore_index=True
    )


# ------------------------------------------------------------
# Save cleaned groundwater dataset
# ------------------------------------------------------------

groundwater_output = (
    PROCESSED_DATA /
    "west_bengal_groundwater_2024_clean.csv"
)

save_clean_dataset(
    groundwater,
    groundwater_output
)


# ============================================================
# PART 2 - SURFACE PHYSICAL
# ============================================================

print("\n" + "=" * 80)
print("2. SURFACE PHYSICAL CLEANING")
print("=" * 80)


surface_physical_file = (
    PROCESSED_DATA /
    "west_bengal_surface_physical_combined.csv"
)

surface_physical = pd.read_csv(
    surface_physical_file,
    low_memory=False
)

print(
    "Original shape:",
    surface_physical.shape
)


# ------------------------------------------------------------
# Standardize columns
# ------------------------------------------------------------

surface_physical = clean_column_names(
    surface_physical
)


# ------------------------------------------------------------
# Replace common missing markers
# ------------------------------------------------------------

surface_physical = surface_physical.replace(
    {
        "_": pd.NA,
        "NA": pd.NA,
        "N/A": pd.NA,
        "": pd.NA,
        " ": pd.NA
    }
)


# ------------------------------------------------------------
# Convert physical measurements
# ------------------------------------------------------------

surface_physical_numeric = [
    "Electric Conductivity (μS/cm)",
    "Turbidity (NTU)",
    "Temperature (ºC)",
    "Total Solids (mg/L)"
]

surface_physical = convert_numeric_columns(
    surface_physical,
    surface_physical_numeric
)


# ------------------------------------------------------------
# Remove completely empty columns
# ------------------------------------------------------------

surface_physical = remove_empty_columns(
    surface_physical
)


# ------------------------------------------------------------
# Remove exact duplicates
# ------------------------------------------------------------

duplicate_count = surface_physical.duplicated().sum()

print(
    "Duplicate surface physical rows:",
    duplicate_count
)

if duplicate_count > 0:
    surface_physical = surface_physical.drop_duplicates(
        ignore_index=True
    )


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

surface_physical_output = (
    PROCESSED_DATA /
    "west_bengal_surface_physical_clean.csv"
)

save_clean_dataset(
    surface_physical,
    surface_physical_output
)


# ============================================================
# PART 3 - SURFACE BIOLOGICAL
# ============================================================

print("\n" + "=" * 80)
print("3. SURFACE BIOLOGICAL CLEANING")
print("=" * 80)


surface_biological_file = (
    PROCESSED_DATA /
    "west_bengal_surface_biological_combined.csv"
)

surface_biological = pd.read_csv(
    surface_biological_file,
    low_memory=False
)

print(
    "Original shape:",
    surface_biological.shape
)


# ------------------------------------------------------------
# Standardize columns
# ------------------------------------------------------------

surface_biological = clean_column_names(
    surface_biological
)


# ------------------------------------------------------------
# Replace missing markers
# ------------------------------------------------------------

surface_biological = surface_biological.replace(
    {
        "_": pd.NA,
        "NA": pd.NA,
        "N/A": pd.NA,
        "": pd.NA,
        " ": pd.NA
    }
)


# ------------------------------------------------------------
# Convert biological measurements
# ------------------------------------------------------------

surface_biological_numeric = [
    "Biochemical Oxygen Demand (mg/L)",
    "Chemical Oxygen Demand (mg/L)",
    "Fecal Coliform (MPN/100mL)",
    "Total Coliform (MPN/100mL)"
]

surface_biological = convert_numeric_columns(
    surface_biological,
    surface_biological_numeric
)


# ------------------------------------------------------------
# Remove completely empty columns
# ------------------------------------------------------------

surface_biological = remove_empty_columns(
    surface_biological
)


# ------------------------------------------------------------
# Remove exact duplicates
# ------------------------------------------------------------

duplicate_count = surface_biological.duplicated().sum()

print(
    "Duplicate surface biological rows:",
    duplicate_count
)

if duplicate_count > 0:
    surface_biological = surface_biological.drop_duplicates(
        ignore_index=True
    )


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

surface_biological_output = (
    PROCESSED_DATA /
    "west_bengal_surface_biological_clean.csv"
)

save_clean_dataset(
    surface_biological,
    surface_biological_output
)


# ============================================================
# PART 4 - CLEANING SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("CLEANING COMPLETED")
print("=" * 80)

print("\nCleaned datasets created:")

print(
    groundwater_output
)

print(
    surface_physical_output
)

print(
    surface_biological_output
)

print("\nRaw datasets were not modified.")

print("\n" + "=" * 80)