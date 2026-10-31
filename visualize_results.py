
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. Project paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"

COMPARISON_FILE = REPORTS_DIR / "model_comparison.csv"
CONFUSION_FILE = REPORTS_DIR / "confusion_matrix_bilstm.csv"

sns.set_theme(style="whitegrid")


# ==========================================
# 2. Load model comparison data
# ==========================================

if not COMPARISON_FILE.exists():
    raise FileNotFoundError(
        f"Comparison file not found: {COMPARISON_FILE}"
    )

comparison = pd.read_csv(COMPARISON_FILE)

required_columns = [
    "Model",
    "Accuracy",
    "Precision_macro",
    "Recall_macro",
    "F1_macro"
]

missing_columns = [
    column for column in required_columns
    if column not in comparison.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in model_comparison.csv: {missing_columns}"
    )

metrics = [
    "Accuracy",
    "Precision_macro",
    "Recall_macro",
    "F1_macro"
]

# Convert metric values to percentages for display.
for metric in metrics:
    comparison[metric] = pd.to_numeric(
        comparison[metric], errors="raise"
    ) * 100


# ==========================================
# 3. Accuracy comparison chart
# ==========================================

plt.figure(figsize=(10, 6))

ax = sns.barplot(
    data=comparison,
    x="Model",
    y="Accuracy"
)

ax.set_title("Model Accuracy Comparison", fontsize=16)
ax.set_xlabel("Model")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f%%", padding=3)

plt.xticks(rotation=10, ha="right")
plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ==========================================
# 4. Compare precision, recall, and F1-score
# ==========================================

metric_long = comparison.melt(
    id_vars="Model",
    value_vars=[
        "Precision_macro",
        "Recall_macro",
        "F1_macro"
    ],
    var_name="Metric",
    value_name="Score"
)

metric_long["Metric"] = metric_long["Metric"].replace({
    "Precision_macro": "Macro Precision",
    "Recall_macro": "Macro Recall",
    "F1_macro": "Macro F1-score"
})

plt.figure(figsize=(11, 6))

ax = sns.barplot(
    data=metric_long,
    x="Metric",
    y="Score",
    hue="Model"
)

ax.set_title("Model Performance Metrics Comparison", fontsize=16)
ax.set_xlabel("Evaluation Metric")
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 100)

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8)

plt.legend(title="Model")
plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "model_metrics_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ==========================================
# 5. BiLSTM confusion matrix heatmap
# ==========================================

if CONFUSION_FILE.exists():
    confusion_df = pd.read_csv(CONFUSION_FILE)

    # Use numeric columns as the confusion matrix values.
    numeric_df = confusion_df.select_dtypes(include="number")

    if numeric_df.shape == (2, 2):
        matrix = numeric_df.to_numpy()

        plt.figure(figsize=(7, 6))

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted Fake", "Predicted Real"],
            yticklabels=["Actual Fake", "Actual Real"]
        )

        plt.title("BiLSTM Confusion Matrix", fontsize=16)
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.tight_layout()

        plt.savefig(
            REPORTS_DIR / "bilstm_confusion_matrix.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    else:
        print(
            "Confusion matrix format was not recognized. "
            "Check confusion_matrix_bilstm.csv."
        )

else:
    print("Confusion matrix file not found.")


# ==========================================
# 6. Display comparison table
# ==========================================

print("\nModel Comparison")
print("=" * 75)

display_columns = [
    "Model",
    "Accuracy",
    "Precision_macro",
    "Recall_macro",
    "F1_macro"
]

print(
    comparison[display_columns]
    .round(2)
    .to_string(index=False)
)

print("\nCharts saved in:", REPORTS_DIR)
print("1. accuracy_comparison.png")
print("2. model_metrics_comparison.png")

if (REPORTS_DIR / "bilstm_confusion_matrix.png").exists():
    print("3. bilstm_confusion_matrix.png")
