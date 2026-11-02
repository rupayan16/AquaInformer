from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# PATHS
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "groundwater_features.csv"
OUTPUT_FILE = PROCESSED_DIR / "groundwater_with_targets.csv"


# =========================================================
# LOAD DATA
# =========================================================
print("\n" + "=" * 60)
print("AQUAINFORMER - TARGET CREATION")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("Input shape:", df.shape)


# =========================================================
# REFERENCE LIMITS
# =========================================================
# These are project reference values for screening/classification.
#
# They are NOT being presented as disease thresholds.
# They are also NOT a substitute for laboratory certification
# or regulatory compliance testing.
#
# The target is based on the number of available parameters
# exceeding the selected reference limits.
#
# IMPORTANT:
# Keep the reference values documented in the project report
# and cite the chosen drinking-water standard there.
# =========================================================

REFERENCE_LIMITS = {
    "ph": {
        "lower": 6.5,
        "upper": 8.5,
    },

    "tds_mg_l": {
        "upper": 500,
    },

    "cl_mg_l": {
        "upper": 250,
    },

    "no3_mg_l": {
        "upper": 45,
    },

    "so4_mg_l": {
        "upper": 200,
    },

    "f_mg_l": {
        "upper": 1.0,
    },

    "total_hardness_mg_l": {
        "upper": 200,
    },

    "fe_mg_l": {
        "upper": 1.0,
    },

    "as_ppb": {
        "upper": 10,
    },
}


# =========================================================
# PARAMETER VIOLATION FLAGS
# =========================================================
print("\nCreating parameter-level compliance flags...")


def create_violation_flag(series, limits):
    """
    Return:
        0 = within reference range
        1 = exceeds reference range
        NaN = unavailable
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype="float64"
    )

    valid = numeric.notna()

    if "lower" in limits:
        lower_violation = (
            numeric < limits["lower"]
        )

        result.loc[valid] = (
            lower_violation.loc[valid]
            .astype(int)
        )

    if "upper" in limits:
        upper_violation = (
            numeric > limits["upper"]
        )

        if "lower" in limits:
            combined_violation = (
                lower_violation
                | upper_violation
            )

            result.loc[valid] = (
                combined_violation.loc[valid]
                .astype(int)
            )
        else:
            result.loc[valid] = (
                upper_violation.loc[valid]
                .astype(int)
            )

    return result


violation_columns = []

for parameter, limits in REFERENCE_LIMITS.items():

    if parameter not in df.columns:
        print(
            f"Skipping {parameter}: "
            "column not found."
        )
        continue

    flag_column = parameter + "_violation"

    df[flag_column] = create_violation_flag(
        df[parameter],
        limits
    )

    violation_columns.append(flag_column)


# =========================================================
# CALCULATE COMPLIANCE SCORE
# =========================================================
print("\nCalculating compliance score...")

df["parameters_available"] = (
    df[violation_columns]
    .notna()
    .sum(axis=1)
)

df["parameters_exceeded"] = (
    df[violation_columns]
    .fillna(0)
    .sum(axis=1)
)


df["violation_rate"] = np.where(
    df["parameters_available"] > 0,
    df["parameters_exceeded"]
    / df["parameters_available"],
    np.nan
)


# =========================================================
# WATER QUALITY CLASS
# =========================================================
def classify_water_quality(row):

    available = row["parameters_available"]
    exceeded = row["parameters_exceeded"]

    if available == 0:
        return np.nan

    # No detected parameter exceedances
    if exceeded == 0:
        return "Good"

    # One exceedance
    if exceeded <= 1:
        return "Moderate"

    # Two or more exceedances
    return "Poor"


df["water_quality_class"] = df.apply(
    classify_water_quality,
    axis=1
)


# =========================================================
# CONTAMINATION RISK
# =========================================================
def classify_contamination_risk(row):

    available = row["parameters_available"]
    rate = row["violation_rate"]

    if available == 0 or pd.isna(rate):
        return np.nan

    if rate == 0:
        return "Low"

    if rate <= 0.25:
        return "Moderate"

    return "High"


df["contamination_risk"] = df.apply(
    classify_contamination_risk,
    axis=1
)


# =========================================================
# REMOVE ROWS WITHOUT A TARGET
# =========================================================
df_target = df[
    df["water_quality_class"].notna()
    & df["contamination_risk"].notna()
].copy()


# =========================================================
# SUMMARY
# =========================================================
print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print("\nWater Quality Class:")
print(
    df_target["water_quality_class"]
    .value_counts()
)

print("\nContamination Risk:")
print(
    df_target["contamination_risk"]
    .value_counts()
)


print("\nParameter violation counts:")

for column in violation_columns:

    parameter = column.replace(
        "_violation",
        ""
    )

    count = int(
        df[column]
        .fillna(0)
        .sum()
    )

    print(
        f"{parameter}: {count}"
    )


# =========================================================
# SAVE
# =========================================================
df_target.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\nFinal dataset shape:")
print(df_target.shape)

print("\n" + "=" * 60)
print("TARGET CREATION COMPLETED")
print("=" * 60)