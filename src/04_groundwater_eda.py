import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# AquaInformer - Step 4: Groundwater Exploratory Data Analysis
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = PROJECT_ROOT / "data" / "raw"
VISUALIZATION = PROJECT_ROOT / "visualization"
REPORTS = PROJECT_ROOT / "reports"

VISUALIZATION.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Load groundwater dataset
# ------------------------------------------------------------

file_path = RAW_DATA / "west_bengal_gwq_2024.csv"

df = pd.read_csv(
    file_path,
    low_memory=False
)

print("=" * 80)
print("AQUAINFORMER - GROUNDWATER EDA")
print("=" * 80)

print("\nDataset shape:")
print(df.shape)


# ------------------------------------------------------------
# Standardize column names
# ------------------------------------------------------------

df.columns = (
    df.columns
    .str.replace("\n", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

print("\nColumns after formatting:")
for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")


# ------------------------------------------------------------
# Important groundwater parameters
# ------------------------------------------------------------

parameter_columns = [
    "pH",
    "EC (µ S-cm at 25 ° C)",
    "TDS (mg/L)",
    "Total Alkalinity (mg/L)",
    "Cl (mg/L)",
    "NO3 (mg/L)",
    "SO4 (mg/L)",
    "F (mg/L)",
    "Total Hardness (mg/L)",
    "Ca (mg/L)",
    "Mg (mg/L)",
    "Na (mg/L)",
    "K (mg/L)",
    "Fe (mg/L)",
    "As (ppb)"
]


# ------------------------------------------------------------
# Convert selected parameters to numeric
# ------------------------------------------------------------

for column in parameter_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ------------------------------------------------------------
# Numeric summary
# ------------------------------------------------------------

numeric_df = df[[
    column
    for column in parameter_columns
    if column in df.columns
]]

print("\n" + "=" * 80)
print("NUMERIC SUMMARY")
print("=" * 80)

print(
    numeric_df.describe().T
)


# ------------------------------------------------------------
# Missing-value analysis
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)

missing = pd.DataFrame({
    "Missing_Count": numeric_df.isnull().sum(),
    "Missing_Percentage":
        numeric_df.isnull().mean() * 100
})

print(
    missing.sort_values(
        "Missing_Percentage",
        ascending=False
    )
)


# ------------------------------------------------------------
# Save summary
# ------------------------------------------------------------

summary_file = REPORTS / "groundwater_numeric_summary.csv"

numeric_df.describe().T.to_csv(
    summary_file
)

missing_file = REPORTS / "groundwater_missing_summary.csv"

missing.to_csv(
    missing_file
)


# ------------------------------------------------------------
# Histograms
# ------------------------------------------------------------

print("\nCreating histograms...")

for column in numeric_df.columns:

    data = numeric_df[column].dropna()

    if data.empty:
        continue

    plt.figure(figsize=(8, 5))

    plt.hist(
        data,
        bins=30
    )

    plt.title(
        f"Distribution of {column}"
    )

    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()

    safe_name = (
        column
        .replace("/", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("°", "")
        .replace("µ", "u")
    )

    output_file = (
        VISUALIZATION
        / f"groundwater_hist_{safe_name}.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()


# ------------------------------------------------------------
# Correlation matrix
# ------------------------------------------------------------

print("\nCreating correlation matrix...")

correlation = numeric_df.corr(
    numeric_only=True
)

print("\nCorrelation matrix:")
print(correlation.round(2))


correlation_file = (
    REPORTS
    / "groundwater_correlation.csv"
)

correlation.to_csv(
    correlation_file
)


# ------------------------------------------------------------
# Correlation heatmap
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 10)
)

plt.imshow(
    correlation,
    aspect="auto"
)

plt.colorbar(
    label="Correlation"
)

plt.xticks(
    range(len(correlation.columns)),
    list(correlation.columns),
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    list(correlation.columns)
)

plt.title(
    "Groundwater Parameter Correlation Matrix"
)

plt.tight_layout()

heatmap_file = (
    VISUALIZATION
    / "groundwater_correlation_heatmap.png"
)

plt.savefig(
    heatmap_file,
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Boxplots
# ------------------------------------------------------------

print("\nCreating boxplots...")

for column in numeric_df.columns:

    data = numeric_df[column].dropna()

    if data.empty:
        continue

    plt.figure(figsize=(8, 4))

    plt.boxplot(
        data,
        vert=False
    )

    plt.title(
        f"Boxplot of {column}"
    )

    plt.xlabel(column)

    plt.tight_layout()

    safe_name = (
        column
        .replace("/", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("°", "")
        .replace("µ", "u")
    )

    output_file = (
        VISUALIZATION
        / f"groundwater_boxplot_{safe_name}.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()


# ------------------------------------------------------------
# Completion message
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("GROUNDWATER EDA COMPLETED")
print("=" * 80)

print("\nReports created:")
print(summary_file)
print(missing_file)
print(correlation_file)

print("\nVisualizations saved in:")
print(VISUALIZATION)