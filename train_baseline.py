
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ==========================================
# 1. Project paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "WELFake_cleaned.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_PATH = MODEL_DIR / "tfidf_logistic_regression.joblib"

# Automatically create folders if they do not exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. Load cleaned dataset
# ==========================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

# Check required columns
required_columns = {"content", "label"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        "Dataset must contain 'content' and 'label' columns."
    )

# Remove missing values and empty news text
df = df.dropna(subset=["content", "label"]).copy()
df["content"] = df["content"].astype(str).str.strip()
df = df[df["content"] != ""].copy()

# Ensure labels are numeric
df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

# Expected WELFake label mapping:
# 0 = Fake News, 1 = Real News
if not set(df["label"].unique()).issubset({0, 1}):
    raise ValueError("Labels must be 0 (Fake) or 1 (Real).")

X = df["content"]
y = df["label"]

print(f"Total samples: {len(df)}")
print("\nLabel distribution:")
print(y.value_counts().sort_index().rename(index={
    0: "Fake News",
    1: "Real News"
}))


# ==========================================
# 3. Split dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ==========================================
# 4. Build and train model
# ==========================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=50000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
    ),
])

print("\nTraining model...")
model.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# 5. Predict test dataset
# ==========================================

print("\nPredicting test samples...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test Accuracy (%): {accuracy * 100:.2f}%")


# ==========================================
# 6. Create test predictions CSV
# ==========================================

label_names = {
    0: "Fake News",
    1: "Real News",
}

# y_test and X_test retain their original dataframe indices.
# Using those indices keeps each text aligned with its label.
predictions_df = pd.DataFrame({
    "original_index": X_test.index,
    "news_content": X_test.to_numpy(),
    "actual_label": y_test.to_numpy(),
    "predicted_label": y_pred,
    "actual_class": [
        label_names[int(label)] for label in y_test
    ],
    "predicted_class": [
        label_names[int(label)] for label in y_pred
    ],
    "fake_probability": y_prob[:, 0],
    "real_probability": y_prob[:, 1],
})

predictions_df["correct_prediction"] = (
    predictions_df["actual_label"]
    == predictions_df["predicted_label"]
)

predictions_path = REPORT_DIR / "test_predictions.csv"

predictions_df.to_csv(
    predictions_path,
    index=False,
    encoding="utf-8-sig",
)

print(f"\nTest predictions saved to:\n{predictions_path}")


# ==========================================
# 7. Save misclassified examples
# ==========================================

errors_df = predictions_df[
    predictions_df["correct_prediction"] == False
].copy()

errors_path = REPORT_DIR / "misclassified_examples.csv"

errors_df.to_csv(
    errors_path,
    index=False,
    encoding="utf-8-sig",
)

print(f"Incorrect predictions: {len(errors_df)}")
print(f"Misclassified examples saved to:\n{errors_path}")


# ==========================================
# 8. Save classification report
# ==========================================

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Fake News", "Real News"],
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report).transpose()

report_path = REPORT_DIR / "classification_report.csv"

report_df.to_csv(
    report_path,
    encoding="utf-8-sig",
)

print(f"Classification report saved to:\n{report_path}")


# ==========================================
# 9. Print confusion matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

cm_df = pd.DataFrame(
    cm,
    index=["Actual Fake", "Actual Real"],
    columns=["Predicted Fake", "Predicted Real"],
)

print("\nConfusion Matrix:")
print(cm_df)


# ==========================================
# 10. Save trained model
# ==========================================

joblib.dump(model, MODEL_PATH)

print(f"\nTrained model saved to:\n{MODEL_PATH}")

print("\nAll tasks completed successfully!")
