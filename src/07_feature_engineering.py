from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


GROUNDWATER_INPUT = (
    PROCESSED_DIR / "west_bengal_groundwater_2024_clean.csv"
)

SURFACE_PHYSICAL_INPUT = (
    PROCESSED_DIR / "west_bengal_surface_physical_clean.csv"
)

SURFACE_BIOLOGICAL_INPUT = (
    PROCESSED_DIR / "west_bengal_surface_biological_clean.csv"
)


GROUNDWATER_OUTPUT = (
    PROCESSED_DIR / "groundwater_features.csv"
)

SURFACE_OUTPUT = (
    PROCESSED_DIR / "surface_features.csv"
)


# ---------------------------------------------------------
# HELPER FUNCTION
# ---------------------------------------------------------
def clean_feature_names(df):
    """Standardize column names for ML processing."""
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    return df


# ---------------------------------------------------------
# GROUNDWATER FEATURES
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("GROUNDWATER FEATURE ENGINEERING")
print("=" * 60)

groundwater = pd.read_csv(GROUNDWATER_INPUT)

groundwater = clean_feature_names(groundwater)

print("Groundwater shape:", groundwater.shape)


# Candidate groundwater water-quality parameters
groundwater_features = [
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


available_groundwater_features = [
    column
    for column in groundwater_features
    if column in groundwater.columns
]

groundwater_ml = groundwater[
    available_groundwater_features
].copy()


# Convert all candidate features to numeric
for column in groundwater_ml.columns:
    groundwater_ml[column] = pd.to_numeric(
        groundwater_ml[column],
        errors="coerce"
    )


# Remove columns that contain no usable values
groundwater_ml = groundwater_ml.dropna(
    axis=1,
    how="all"
)


print("\nGroundwater ML features:")
for column in groundwater_ml.columns:
    print(" -", column)


print("\nGroundwater feature missing values:")
print(groundwater_ml.isna().sum())


# Save
groundwater_ml.to_csv(
    GROUNDWATER_OUTPUT,
    index=False
)


print(
    "\nSaved:",
    GROUNDWATER_OUTPUT
)


# ---------------------------------------------------------
# SURFACE FEATURES
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("SURFACE WATER FEATURE ENGINEERING")
print("=" * 60)

surface_physical = pd.read_csv(
    SURFACE_PHYSICAL_INPUT
)

surface_biological = pd.read_csv(
    SURFACE_BIOLOGICAL_INPUT
)

surface_physical = clean_feature_names(
    surface_physical
)

surface_biological = clean_feature_names(
    surface_biological
)


print(
    "Surface physical shape:",
    surface_physical.shape
)

print(
    "Surface biological shape:",
    surface_biological.shape
)


# Candidate physical parameters
physical_features = [
    "electric_conductivity_s_cm",
    "turbidity_ntu",
    "total_solids_mg_l",
]


# Candidate biological parameters
biological_features = [
    "biochemical_oxygen_demand_mg_l",
    "fecal_coliform_mpn_100ml",
    "total_coliform_mpn_100ml",
]


available_physical = [
    column
    for column in physical_features
    if column in surface_physical.columns
]

available_biological = [
    column
    for column in biological_features
    if column in surface_biological.columns
]


physical = surface_physical[
    available_physical
].copy()

biological = surface_biological[
    available_biological
].copy()


# Convert numeric columns
for column in physical.columns:
    physical[column] = pd.to_numeric(
        physical[column],
        errors="coerce"
    )

for column in biological.columns:
    biological[column] = pd.to_numeric(
        biological[column],
        errors="coerce"
    )


# Combine physical and biological features by row.
# The two source files originate from the same CPCB
# station/time structure, but we do not assume perfect
# row-level matching for ML at this stage.
surface_features = pd.concat(
    [physical.reset_index(drop=True),
     biological.reset_index(drop=True)],
    axis=1
)


# Remove completely empty columns
surface_features = surface_features.dropna(
    axis=1,
    how="all"
)


print("\nSurface ML features:")
for column in surface_features.columns:
    print(" -", column)


print("\nSurface feature missing values:")
print(surface_features.isna().sum())


# Save
surface_features.to_csv(
    SURFACE_OUTPUT,
    index=False
)


print(
    "\nSaved:",
    SURFACE_OUTPUT
)


# ---------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print(
    "Groundwater features:",
    groundwater_ml.shape
)

print(
    "Surface features:",
    surface_features.shape
)

print("\nNext step: create standards-based targets.")