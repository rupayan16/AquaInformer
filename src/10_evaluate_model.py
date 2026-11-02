from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)


# =========================================================
# PATHS
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEST_FILE = (
    PROCESSED_DIR / "test_set.csv"
)

QUALITY_MODEL_FILE = (
    MODEL_DIR / "water_quality_model.joblib"
)

RISK_MODEL_FILE = (
    MODEL_DIR / "contamination_risk_model.joblib"
)


# =========================================================
# LOAD
# =========================================================
print("\n" + "=" * 60)
print("AQUAINFORMER - MODEL EVALUATION")
print("=" * 60)

test = pd.read_csv(TEST_FILE)

quality_model = joblib.load(
    QUALITY_MODEL_FILE
)

risk_model = joblib.load(
    RISK_MODEL_FILE
)


# =========================================================
# FEATURES
# =========================================================
target_columns = [
    "actual_water_quality",
    "actual_contamination_risk",
]

feature_columns = [
    column
    for column in test.columns
    if column not in target_columns
]


X_test = test[feature_columns]

y_quality = test[
    "actual_water_quality"
]

y_risk = test[
    "actual_contamination_risk"
]


# =========================================================
# PREDICTIONS
# =========================================================
print("\nGenerating predictions...")

quality_predictions = quality_model.predict(
    X_test
)

risk_predictions = risk_model.predict(
    X_test
)


# =========================================================
# EVALUATION FUNCTION
# =========================================================
def evaluate_model(
    model_name,
    y_true,
    y_pred,
    output_prefix
):

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred
        )
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )


    print(f"\nAccuracy:            {accuracy:.4f}")
    print(
        f"Balanced Accuracy:  {balanced_accuracy:.4f}"
    )
    print(
        f"Weighted Precision:  {precision:.4f}"
    )
    print(
        f"Weighted Recall:     {recall:.4f}"
    )
    print(
        f"Weighted F1:         {f1:.4f}"
    )
    print(
        f"Macro F1:            {macro_f1:.4f}"
    )


    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------
    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )


    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_file = (
        REPORT_DIR
        / f"{output_prefix}_classification_report.csv"
    )

    report_df.to_csv(
        report_file
    )


    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------
    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("Confusion Matrix:")
    print(cm)


    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=sorted(
            y_true.unique()
        )
    )

    display.plot(
        xticks_rotation="vertical"
    )

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    plt.tight_layout()


    figure_file = (
        REPORT_DIR
        / f"{output_prefix}_confusion_matrix.png"
    )

    plt.savefig(
        figure_file,
        dpi=300
    )

    plt.close()


    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced_accuracy,
        "Weighted_Precision": precision,
        "Weighted_Recall": recall,
        "Weighted_F1": f1,
        "Macro_F1": macro_f1,
    }


# =========================================================
# EVALUATE BOTH MODELS
# =========================================================
quality_results = evaluate_model(
    "Water Quality Classification",
    y_quality,
    quality_predictions,
    "water_quality"
)


risk_results = evaluate_model(
    "Contamination Risk Classification",
    y_risk,
    risk_predictions,
    "contamination_risk"
)


# =========================================================
# SAVE SUMMARY
# =========================================================
results = pd.DataFrame(
    [
        quality_results,
        risk_results,
    ]
)

summary_file = (
    REPORT_DIR / "model_evaluation_summary.csv"
)

results.to_csv(
    summary_file,
    index=False
)


# =========================================================
# SAVE PREDICTIONS
# =========================================================
prediction_output = X_test.copy()

prediction_output[
    "actual_water_quality"
] = y_quality.values

prediction_output[
    "predicted_water_quality"
] = quality_predictions

prediction_output[
    "actual_contamination_risk"
] = y_risk.values

prediction_output[
    "predicted_contamination_risk"
] = risk_predictions


prediction_file = (
    PROCESSED_DIR
    / "model_predictions.csv"
)

prediction_output.to_csv(
    prediction_file,
    index=False
)


# =========================================================
# FINAL OUTPUT
# =========================================================
print("\n" + "=" * 60)
print("MODEL EVALUATION COMPLETED")
print("=" * 60)

print("\nEvaluation summary:")
print(results.to_string(index=False))

print("\nSaved:")
print(summary_file)

print("\nSaved:")
print(prediction_file)

print("\nConfusion matrices and classification reports")
print("are available in the reports folder.")