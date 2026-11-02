import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# AquaInformer - Step 5: Surface Water EDA
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
VISUALIZATION = PROJECT_ROOT / "visualization"
REPORTS = PROJECT_ROOT / "reports"

VISUALIZATION.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper function
# ============================================================

def load_csv(file_path):
    """Load a CSV file and return a DataFrame."""
    return pd.read_csv(
        file_path,
        low_memory=False
    )


# ============================================================
# 1. Load surface physical data
# ============================================================

physical_file = (
    PROCESSED_DATA /
    "west_bengal_surface_physical_combined.csv"
)

physical_df = load_csv(physical_file)


# ============================================================
# 2. Load surface biological data
# ============================================================

biological_1961_2020_file = (
    RAW_DATA /
    "swq_manual_biological_parameters_cpcb_wb_1961_2020.csv"
)

biological_2021_2025_file = (
    RAW_DATA /
    "swq_biological_parameter_manual_cpcb_wb_2021_2025.csv"
)


biological_1961_2020 = load_csv(
    biological_1961_2020_file
)

biological_2021_2025 = load_csv(
    biological_2021_2025_file
)


# ============================================================
# 3. Basic information
# ============================================================

print("=" * 80)
print("AQUAINFORMER - SURFACE WATER EDA")
print("=" * 80)

print("\n--- SURFACE PHYSICAL DATA ---")
print("Shape:", physical_df.shape)

print("\n--- SURFACE BIOLOGICAL 1961-2020 ---")
print("Shape:", biological_1961_2020.shape)

print("\n--- SURFACE BIOLOGICAL 2021-2025 ---")
print("Shape:", biological_2021_2025.shape)


# ============================================================
# 4. Standardize column names
# ============================================================

def clean_column_names(df):

    df.columns = (
        df.columns
        .str.replace("\n", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return df


physical_df = clean_column_names(physical_df)
biological_1961_2020 = clean_column_names(
    biological_1961_2020
)
biological_2021_2025 = clean_column_names(
    biological_2021_2025
)


# ============================================================
# 5. Convert important physical parameters
# ============================================================

physical_parameters = [
    "Electric Conductivity (μS/cm)",
    "Turbidity (NTU)",
    "Temperature (ºC)",
    "Total Solids (mg/L)"
]

for column in physical_parameters:

    if column in physical_df.columns:

        physical_df[column] = pd.to_numeric(
            physical_df[column],
            errors="coerce"
        )


# ============================================================
# 6. Physical data summary
# ============================================================

print("\n" + "=" * 80)
print("SURFACE PHYSICAL NUMERIC SUMMARY")
print("=" * 80)

physical_numeric = physical_df.select_dtypes(
    include="number"
)

print(
    physical_numeric.describe().T
)


# ============================================================
# 7. Physical missing-value analysis
# ============================================================

print("\n" + "=" * 80)
print("SURFACE PHYSICAL MISSING VALUES")
print("=" * 80)

physical_missing = pd.DataFrame({
    "Missing_Count": physical_df.isnull().sum(),
    "Missing_Percentage":
        physical_df.isnull().mean() * 100
})

print(
    physical_missing
    .sort_values(
        "Missing_Percentage",
        ascending=False
    )
)


physical_missing.to_csv(
    REPORTS / "surface_physical_missing_summary.csv"
)


# ============================================================
# 8. Physical parameter histograms
# ============================================================

print("\nCreating physical parameter histograms...")

for column in physical_parameters:

    if column not in physical_df.columns:
        continue

    data = physical_df[column].dropna()

    if data.empty:
        continue

    plt.figure(figsize=(8, 5))

    plt.hist(
        data,
        bins=30
    )

    plt.title(
        f"Surface Water - {column}"
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
        .replace("μ", "u")
        .replace("º", "")
    )

    plt.savefig(
        VISUALIZATION /
        f"surface_physical_hist_{safe_name}.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 9. Combine biological datasets
# ============================================================

biological_1961_2020["Dataset_Period"] = "1961-2020"
biological_2021_2025["Dataset_Period"] = "2021-2025"

biological_df = pd.concat(
    [
        biological_1961_2020,
        biological_2021_2025
    ],
    ignore_index=True
)


print("\n" + "=" * 80)
print("COMBINED SURFACE BIOLOGICAL DATA")
print("=" * 80)

print("Shape:", biological_df.shape)


# ============================================================
# 10. Convert biological parameters
# ============================================================

biological_parameters = [
    "Biochemical Oxygen Demand (mg/L)",
    "Chemical Oxygen Demand (mg/L)",
    "Fecal Coliform (MPN/100mL)",
    "Total Coliform (MPN/100mL)"
]

for column in biological_parameters:

    if column in biological_df.columns:

        biological_df[column] = pd.to_numeric(
            biological_df[column],
            errors="coerce"
        )


# ============================================================
# 11. Biological summary
# ============================================================

print("\n" + "=" * 80)
print("SURFACE BIOLOGICAL NUMERIC SUMMARY")
print("=" * 80)

biological_numeric = biological_df.select_dtypes(
    include="number"
)

print(
    biological_numeric.describe().T
)


# ============================================================
# 12. Biological missing values
# ============================================================

print("\n" + "=" * 80)
print("SURFACE BIOLOGICAL MISSING VALUES")
print("=" * 80)

biological_missing = pd.DataFrame({
    "Missing_Count": biological_df.isnull().sum(),
    "Missing_Percentage":
        biological_df.isnull().mean() * 100
})

print(
    biological_missing
    .sort_values(
        "Missing_Percentage",
        ascending=False
    )
)


biological_missing.to_csv(
    REPORTS / "surface_biological_missing_summary.csv"
)


# ============================================================
# 13. Biological histograms
# ============================================================

print("\nCreating biological parameter histograms...")

for column in biological_parameters:

    if column not in biological_df.columns:
        continue

    data = biological_df[column].dropna()

    if data.empty:
        continue

    plt.figure(figsize=(8, 5))

    plt.hist(
        data,
        bins=30
    )

    plt.title(
        f"Surface Water - {column}"
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
    )

    plt.savefig(
        VISUALIZATION /
        f"surface_biological_hist_{safe_name}.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 14. Log-transformed biological distributions
# ============================================================

log_parameters = [
    "Fecal Coliform (MPN/100mL)",
    "Total Coliform (MPN/100mL)"
]

print("\nCreating log-scale biological histograms...")

for column in log_parameters:

    if column not in biological_df.columns:
        continue

    data = biological_df[column].dropna()

    # Keep only positive values for log10
    data = data[data > 0]

    if data.empty:
        continue

    log_data = data.apply(
        lambda x: __import__("math").log10(x)
    )

    plt.figure(figsize=(8, 5))

    plt.hist(
        log_data,
        bins=30
    )

    plt.title(
        f"Log10 Distribution of {column}"
    )

    plt.xlabel(
        f"log10({column})"
    )

    plt.ylabel("Frequency")

    plt.tight_layout()

    safe_name = (
        column
        .replace("/", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    plt.savefig(
        VISUALIZATION /
        f"surface_biological_log_hist_{safe_name}.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 15. Biological correlation
# ============================================================

biological_correlation = biological_numeric.corr()

print("\n" + "=" * 80)
print("SURFACE BIOLOGICAL CORRELATION")
print("=" * 80)

print(
    biological_correlation.round(2)
)


biological_correlation.to_csv(
    REPORTS / "surface_biological_correlation.csv"
)


# ============================================================
# 16. BOD vs Fecal Coliform
# ============================================================

bod_column = "Biochemical Oxygen Demand (mg/L)"
fecal_column = "Fecal Coliform (MPN/100mL)"

if (
    bod_column in biological_df.columns
    and fecal_column in biological_df.columns
):

    scatter_df = biological_df[
        [bod_column, fecal_column]
    ].dropna()

    if not scatter_df.empty:

        plt.figure(figsize=(8, 6))

        plt.scatter(
            scatter_df[bod_column],
            scatter_df[fecal_column],
            alpha=0.5
        )

        plt.title(
            "BOD vs Fecal Coliform"
        )

        plt.xlabel(
            "Biochemical Oxygen Demand (mg/L)"
        )

        plt.ylabel(
            "Fecal Coliform (MPN/100mL)"
        )

        plt.tight_layout()

        plt.savefig(
            VISUALIZATION /
            "surface_bod_vs_fecal_coliform.png",
            dpi=300
        )

        plt.close()


# ============================================================
# 17. Log BOD vs Log Fecal Coliform
# ============================================================

if (
    bod_column in biological_df.columns
    and fecal_column in biological_df.columns
):

    log_scatter = biological_df[
        [bod_column, fecal_column]
    ].dropna()

    log_scatter = log_scatter[
        (log_scatter[bod_column] > 0)
        &
        (log_scatter[fecal_column] > 0)
    ].copy()

    if not log_scatter.empty:

        log_scatter["log_BOD"] = (
            log_scatter[bod_column]
            .apply(lambda x: __import__("math").log10(x))
        )

        log_scatter["log_Fecal_Coliform"] = (
            log_scatter[fecal_column]
            .apply(lambda x: __import__("math").log10(x))
        )

        plt.figure(figsize=(8, 6))

        plt.scatter(
            log_scatter["log_BOD"],
            log_scatter["log_Fecal_Coliform"],
            alpha=0.5
        )

        plt.title(
            "Log10 BOD vs Log10 Fecal Coliform"
        )

        plt.xlabel(
            "log10(BOD)"
        )

        plt.ylabel(
            "log10(Fecal Coliform)"
        )

        plt.tight_layout()

        plt.savefig(
            VISUALIZATION /
            "surface_log_bod_vs_log_fecal_coliform.png",
            dpi=300
        )

        plt.close()


# ============================================================
# 18. Save combined biological dataset
# ============================================================

biological_output = (
    PROCESSED_DATA /
    "west_bengal_surface_biological_combined.csv"
)

biological_df.to_csv(
    biological_output,
    index=False
)


# ============================================================
# Final message
# ============================================================

print("\n" + "=" * 80)
print("SURFACE WATER EDA COMPLETED")
print("=" * 80)

print("\nCombined biological dataset:")
print(biological_output)

print("\nReports saved in:")
print(REPORTS)

print("\nVisualizations saved in:")
print(VISUALIZATION)