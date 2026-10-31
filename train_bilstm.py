
from pathlib import Path
from collections import Counter
import json
import re
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ==================================================
# 1. Settings and reproducibility
# ==================================================

SEED = 42
MAX_VOCAB_SIZE = 30000
MAX_LENGTH = 250
EMBEDDING_DIM = 128
HIDDEN_DIM = 64
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
PATIENCE = 2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "WELFake_cleaned.csv"
BASELINE_PATH = BASE_DIR / "models" / "tfidf_logistic_regression.joblib"

MODEL_DIR = BASE_DIR / "models" / "bilstm"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "bilstm_model.pth"
VOCAB_PATH = MODEL_DIR / "vocab.json"
CONFIG_PATH = MODEL_DIR / "config.json"

print(f"Device: {DEVICE}")


# ==================================================
# 2. Load cleaned dataset
# ==================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

required_columns = {"content", "label"}
if not required_columns.issubset(df.columns):
    raise ValueError(
        "Dataset must contain 'content' and 'label' columns."
    )

df = df.dropna(subset=["content", "label"]).copy()
df["content"] = df["content"].astype(str).str.strip()
df = df[df["content"] != ""].copy()
df["label"] = pd.to_numeric(
    df["label"], errors="raise"
).astype(int)

if not set(df["label"].unique()).issubset({0, 1}):
    raise ValueError("Labels must be 0 (Fake) or 1 (Real).")

X = df["content"]
y = df["label"]

print(f"Total samples: {len(df)}")
print("Label mapping: 0 = Fake News, 1 = Real News")


# ==================================================
# 3. Train / validation / test split
# ==================================================
# Keep the same 20% test split settings as the baseline.
# Validation is taken only from the remaining training data.

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.10,
    random_state=SEED,
    stratify=y_train_full,
)

print(f"\nTraining samples:   {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Testing samples:    {len(X_test)}")


# ==================================================
# 4. Tokenization and vocabulary
# ==================================================

def tokenize(text):
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", str(text).lower())


print("\nBuilding vocabulary from training text...")

counter = Counter()

for text in X_train:
    counter.update(tokenize(text))

# Index 0 is reserved for padding.
# Index 1 is reserved for unknown words.
vocab = {
    "<PAD>": 0,
    "<UNK>": 1,
}

for word, _ in counter.most_common(MAX_VOCAB_SIZE - 2):
    vocab[word] = len(vocab)

print(f"Vocabulary size: {len(vocab)}")


def encode_text(text):
    """Convert a news article to a fixed-length integer sequence."""
    tokens = tokenize(text)

    token_ids = [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens[:MAX_LENGTH]
    ]

    original_length = len(token_ids)

    if original_length == 0:
        token_ids = [vocab["<UNK>"]]
        original_length = 1

    # Pad at the end of the sequence.
    if len(token_ids) < MAX_LENGTH:
        token_ids += [vocab["<PAD>"]] * (
            MAX_LENGTH - len(token_ids)
        )

    return token_ids, original_length


def encode_dataset(texts):
    sequences = []
    lengths = []

    for text in texts:
        sequence, length = encode_text(text)
        sequences.append(sequence)
        lengths.append(length)

    return (
        torch.tensor(sequences, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )


print("Encoding datasets...")

X_train_ids, train_lengths = encode_dataset(X_train)
X_val_ids, val_lengths = encode_dataset(X_val)
X_test_ids, test_lengths = encode_dataset(X_test)

y_train_tensor = torch.tensor(
    y_train.to_numpy(), dtype=torch.float32
)
y_val_tensor = torch.tensor(
    y_val.to_numpy(), dtype=torch.float32
)
y_test_tensor = torch.tensor(
    y_test.to_numpy(), dtype=torch.float32
)


# ==================================================
# 5. PyTorch Dataset and DataLoader
# ==================================================

class NewsDataset(Dataset):
    def __init__(self, sequences, lengths, labels):
        self.sequences = sequences
        self.lengths = lengths
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.sequences[index],
            self.lengths[index],
            self.labels[index],
        )


train_dataset = NewsDataset(
    X_train_ids, train_lengths, y_train_tensor
)
val_dataset = NewsDataset(
    X_val_ids, val_lengths, y_val_tensor
)
test_dataset = NewsDataset(
    X_test_ids, test_lengths, y_test_tensor
)

# num_workers=0 is a simple, Windows-friendly setting.
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ==================================================
# 6. Define the BiLSTM model
# ==================================================

class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        dropout=0.30,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, sequences, lengths):
        embedded = self.embedding(sequences)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden, _) = self.lstm(packed)

        # The last two hidden states represent the
        # forward and backward directions.
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1,
        )

        combined = self.dropout(combined)

        # Return logits; sigmoid is applied for prediction.
        return self.fc(combined).squeeze(1)


model = BiLSTMClassifier(
    vocab_size=len(vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ==================================================
# 7. Training and validation functions
# ==================================================

def run_training_epoch(loader):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for sequences, lengths, labels in loader:
        sequences = sequences.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(sequences, lengths)
        loss = criterion(logits, labels)

        loss.backward()

        # Helps prevent exploding gradients in recurrent models.
        nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        predictions = (logits >= 0).float()
        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples,
    )


def evaluate_loss(loader):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for sequences, lengths, labels in loader:
            sequences = sequences.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(sequences, lengths)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


# ==================================================
# 8. Train model with early stopping
# ==================================================

print("\nStarting BiLSTM training...")

best_val_loss = float("inf")
epochs_without_improvement = 0
best_state = None

for epoch in range(1, EPOCHS + 1):
    train_loss, train_accuracy = run_training_epoch(
        train_loader
    )
    val_loss = evaluate_loss(val_loader)

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy * 100:.2f}% | "
        f"Validation Loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping: validation loss did not improve.")
        break

if best_state is None:
    raise RuntimeError("Training did not produce a model.")

model.load_state_dict(best_state)
model.to(DEVICE)


# ==================================================
# 9. Save model and vocabulary
# ==================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab_size": len(vocab),
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "max_length": MAX_LENGTH,
    },
    MODEL_PATH,
)

with open(VOCAB_PATH, "w", encoding="utf-8") as file:
    json.dump(vocab, file, ensure_ascii=False)

config = {
    "seed": SEED,
    "max_vocab_size": MAX_VOCAB_SIZE,
    "max_length": MAX_LENGTH,
    "embedding_dim": EMBEDDING_DIM,
    "hidden_dim": HIDDEN_DIM,
    "batch_size": BATCH_SIZE,
    "epochs_max": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "label_mapping": {
        "0": "Fake News",
        "1": "Real News",
    },
}

with open(CONFIG_PATH, "w", encoding="utf-8") as file:
    json.dump(config, file, indent=2)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Vocabulary saved to: {VOCAB_PATH}")


# ==================================================
# 10. Predict on the held-out test set
# ==================================================

def predict_probabilities(loader):
    model.eval()
    probabilities = []

    with torch.no_grad():
        for sequences, lengths, _ in loader:
            sequences = sequences.to(DEVICE)

            logits = model(sequences, lengths)
            probs = torch.sigmoid(logits)

            probabilities.extend(
                probs.cpu().numpy().tolist()
            )

    return np.asarray(probabilities)


print("\nEvaluating on test set...")

real_probabilities = predict_probabilities(test_loader)
predictions = (real_probabilities >= 0.5).astype(int)
actual_labels = y_test.to_numpy()

accuracy = accuracy_score(actual_labels, predictions)

precision_macro, recall_macro, f1_macro, _ = (
    precision_recall_fscore_support(
        actual_labels,
        predictions,
        average="macro",
        zero_division=0,
    )
)

print(f"\nBiLSTM Test Accuracy: {accuracy * 100:.2f}%")
print(f"Macro Precision:      {precision_macro:.4f}")
print(f"Macro Recall:         {recall_macro:.4f}")
print(f"Macro F1-score:       {f1_macro:.4f}")


# ==================================================
# 11. Save test predictions and errors
# ==================================================

label_names = {
    0: "Fake News",
    1: "Real News",
}

predictions_df = pd.DataFrame({
    "original_index": X_test.index,
    "news_content": X_test.to_numpy(),
    "actual_label": actual_labels,
    "predicted_label": predictions,
    "actual_class": [
        label_names[int(label)] for label in actual_labels
    ],
    "predicted_class": [
        label_names[int(label)] for label in predictions
    ],
    "fake_probability": 1.0 - real_probabilities,
    "real_probability": real_probabilities,
})

predictions_df["correct_prediction"] = (
    predictions_df["actual_label"]
    == predictions_df["predicted_label"]
)

predictions_path = REPORT_DIR / "test_predictions_bilstm.csv"
errors_path = REPORT_DIR / "misclassified_examples_bilstm.csv"

predictions_df.to_csv(
    predictions_path,
    index=False,
    encoding="utf-8-sig",
)

predictions_df[
    ~predictions_df["correct_prediction"]
].to_csv(
    errors_path,
    index=False,
    encoding="utf-8-sig",
)

print(f"Predictions saved to: {predictions_path}")
print(
    "Incorrect predictions:",
    int((~predictions_df["correct_prediction"]).sum()),
)
print(f"Errors saved to: {errors_path}")


# ==================================================
# 12. Save classification report and confusion matrix
# ==================================================

report = classification_report(
    actual_labels,
    predictions,
    labels=[0, 1],
    target_names=["Fake News", "Real News"],
    output_dict=True,
    zero_division=0,
)

report_path = REPORT_DIR / "classification_report_bilstm.csv"

pd.DataFrame(report).transpose().to_csv(
    report_path,
    encoding="utf-8-sig",
)

cm = confusion_matrix(
    actual_labels, predictions, labels=[0, 1]
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Fake", "Actual Real"],
    columns=["Predicted Fake", "Predicted Real"],
)

cm_path = REPORT_DIR / "confusion_matrix_bilstm.csv"
cm_df.to_csv(cm_path, encoding="utf-8-sig")

print("\nBiLSTM Confusion Matrix:")
print(cm_df)


# ==================================================
# 13. Compare against the Logistic Regression baseline
# ==================================================

comparison_rows = [
    {
        "Model": "BiLSTM",
        "Accuracy": accuracy,
        "Precision_macro": precision_macro,
        "Recall_macro": recall_macro,
        "F1_macro": f1_macro,
    }
]

if BASELINE_PATH.exists():
    print("\nEvaluating saved Logistic Regression baseline...")

    baseline_model = joblib.load(BASELINE_PATH)

    # Same X_test and y_test as the baseline split.
    baseline_predictions = baseline_model.predict(X_test)

    baseline_accuracy = accuracy_score(
        actual_labels, baseline_predictions
    )

    baseline_precision, baseline_recall, baseline_f1, _ = (
        precision_recall_fscore_support(
            actual_labels,
            baseline_predictions,
            average="macro",
            zero_division=0,
        )
    )

    comparison_rows.insert(
        0,
        {
            "Model": "TF-IDF + Logistic Regression",
            "Accuracy": baseline_accuracy,
            "Precision_macro": baseline_precision,
            "Recall_macro": baseline_recall,
            "F1_macro": baseline_f1,
        },
    )

    print(
        f"Baseline Test Accuracy: "
        f"{baseline_accuracy * 100:.2f}%"
    )
else:
    print(
        "Baseline model not found. "
        "Only BiLSTM metrics will be saved."
    )

comparison_df = pd.DataFrame(comparison_rows)
comparison_path = REPORT_DIR / "model_comparison.csv"

comparison_df.to_csv(
    comparison_path,
    index=False,
    encoding="utf-8-sig",
)

print("\nModel comparison:")
print(comparison_df.to_string(index=False))

print(f"\nComparison saved to: {comparison_path}")
print("\nAll BiLSTM tasks completed successfully!")
