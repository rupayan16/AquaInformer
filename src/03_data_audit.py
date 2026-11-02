import pandas as pd
from pathlib import Path


# ============================================================
# AquaInformer - Step 3: Data Audit
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"

REPORTS.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Find files safely
# ------------------------------------------------------------

def find_file(folder, filename):
    """
    Find a file using its exact name first.
    If it does not exist, search for a matching CSV.
    """
    exact_file = folder / filename

    if exact_file.exists():
        return exact_file

    matches = list(folder.glob(filename.replace(".csv", "*.csv")))

    if matches:
        return matches[0]

    return None


# ------------------------------------------------------------
# Dataset files
# ------------------------------------------------------------

groundwater_file = find_file(
    RAW_DATA,
    "west_bengal_gwq_2024.csv"
)

surface_physical_file = (
    PROCESSED_DATA /
    "west_bengal_surface_physical_combined.csv"
)

biological_1961_2020 = find_file(
    RAW_DATA,
    "swq_manual_biological_parameters_cpcb_wb_1961_2020.csv"
)

biological_2021_2025 = find_file(
    RAW_DATA,
    "swq_biological_parameter_manual_cpcb_wb_2021_2025.csv"
)


datasets = {
    "Groundwater 2024": groundwater_file,
    "Surface Physical Combined": surface_physical_file,
    "Surface Biological 1961-2020": biological_1961_2020,
    "Surface Biological 2021-2025": biological_2021_2025,
}


# ------------------------------------------------------------
# Storage for audit results
# ------------------------------------------------------------

data_audit = []
numeric_summary = []
dataset_summary = []


print("=" * 80)
print("AQUAINFORMER - DATA AUDIT")
print("=" * 80)


# ------------------------------------------------------------
# Audit each dataset
# ------------------------------------------------------------

for dataset_name, file_path in datasets.items():

    print("\n" + "=" * 80)
    print(f"DATASET: {dataset_name}")
    print("=" * 80)

    if file_path is None or not file_path.exists():

        print("FILE NOT FOUND")

        dataset_summary.append({
            "Dataset": dataset_name,
            "Status": "File Not Found",
            "Rows": None,
            "Columns": None
        })

        continue

    print("File:", file_path)

    df = pd.read_csv(
        file_path,
        low_memory=False
    )

    rows, columns = df.shape

    print("Rows:", rows)
    print("Columns:", columns)

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    dataset_summary.append({
        "Dataset": dataset_name,
        "Status": "Loaded",
        "Rows": rows,
        "Columns": columns
    })

    # --------------------------------------------------------
    # Missing-value audit
    # --------------------------------------------------------

    missing_count = df.isnull().sum()

    missing_percentage = (
        missing_count / len(df) * 100
    )

    for column in df.columns:

        data_audit.append({
            "Dataset": dataset_name,
            "Column": column,
            "Data_Type": str(df[column].dtype),
            "Total_Rows": len(df),
            "Missing_Count": int(missing_count[column]),
            "Missing_Percentage": round(
                missing_percentage[column], 2
            ),
            "Unique_Values": int(df[column].nunique(
                dropna=True
            ))
        })

    # --------------------------------------------------------
    # Numeric statistics
    # --------------------------------------------------------

    numeric_df = df.select_dtypes(
        include="number"
    )

    if not numeric_df.empty:

        summary = numeric_df.describe().T

        for column in summary.index:

            numeric_summary.append({
                "Dataset": dataset_name,
                "Column": column,
                "Count": summary.loc[column, "count"],
                "Mean": summary.loc[column, "mean"],
                "Std": summary.loc[column, "std"],
                "Min": summary.loc[column, "min"],
                "25%": summary.loc[column, "25%"],
                "Median": summary.loc[column, "50%"],
                "75%": summary.loc[column, "75%"],
                "Max": summary.loc[column, "max"]
            })

    # --------------------------------------------------------
    # Print important missing values
    # --------------------------------------------------------

    print("\nMissing values:")

    missing_display = (
        pd.DataFrame({
            "Missing_Count": missing_count,
            "Missing_Percentage": missing_percentage
        })
        .query("Missing_Count > 0")
        .sort_values(
            "Missing_Percentage",
            ascending=False
        )
    )

    if missing_display.empty:
        print("No missing values found.")

    else:
        print(missing_display)

    print("\n")


# ------------------------------------------------------------
# Convert results to DataFrames
# ------------------------------------------------------------

audit_df = pd.DataFrame(data_audit)

numeric_df = pd.DataFrame(numeric_summary)

dataset_df = pd.DataFrame(dataset_summary)


# ------------------------------------------------------------
# Save Excel audit report
# ------------------------------------------------------------

output_file = REPORTS / "initial_data_audit.xlsx"

with pd.ExcelWriter(
    output_file,
    engine="openpyxl"
) as writer:

    dataset_df.to_excel(
        writer,
        sheet_name="Dataset_Summary",
        index=False
    )

    audit_df.to_excel(
        writer,
        sheet_name="Missing_Value_Audit",
        index=False
    )

    numeric_df.to_excel(
        writer,
        sheet_name="Numeric_Summary",
        index=False
    )


# ------------------------------------------------------------
# Final output
# ------------------------------------------------------------

print("=" * 80)
print("DATA AUDIT COMPLETED")
print("=" * 80)

print("\nAudit report created:")
print(output_file)

print("\nDataset summary:")
print(dataset_df)

print("\n" + "=" * 80)