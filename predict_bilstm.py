
from pathlib import Path
import json
import re

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


# ==========================================
# 1. Project paths and device
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "bilstm" / "bilstm_model.pth"
VOCAB_PATH = BASE_DIR / "models" / "bilstm" / "vocab.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# 2. Define the same BiLSTM architecture
# ==========================================

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

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1,
        )

        combined = self.dropout(combined)

        return self.fc(combined).squeeze(1)


# ==========================================
# 3. Load vocabulary and trained model
# ==========================================

print("Loading BiLSTM model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

if not VOCAB_PATH.exists():
    raise FileNotFoundError(
        f"Vocabulary file not found: {VOCAB_PATH}"
    )

with open(VOCAB_PATH, "r", encoding="utf-8") as file:
    vocab = json.load(file)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

MAX_LENGTH = checkpoint["max_length"]

model = BiLSTMClassifier(
    vocab_size=checkpoint["vocab_size"],
    embedding_dim=checkpoint["embedding_dim"],
    hidden_dim=checkpoint["hidden_dim"],
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print("Model loaded successfully!")
print(f"Device: {DEVICE}")
print(f"Vocabulary size: {len(vocab)}")
print(f"Maximum input length: {MAX_LENGTH} tokens")


# ==========================================
# 4. Tokenize and encode input news
# ==========================================

def tokenize(text):
    """Use the same tokenization as the training script."""
    return re.findall(r"\b\w+\b", str(text).lower())


def encode_text(text):
    tokens = tokenize(text)

    # Keep the first MAX_LENGTH tokens.
    token_ids = [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens[:MAX_LENGTH]
    ]

    length = len(token_ids)

    # Handle input with no recognizable tokens.
    if length == 0:
        token_ids = [vocab["<UNK>"]]
        length = 1

    # Pad sequence to MAX_LENGTH.
    if len(token_ids) < MAX_LENGTH:
        token_ids += [vocab["<PAD>"]] * (
            MAX_LENGTH - len(token_ids)
        )

    sequence_tensor = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    length_tensor = torch.tensor(
        [length],
        dtype=torch.long,
    )

    return sequence_tensor, length_tensor


# ==========================================
# 5. Predict Fake or Real News
# ==========================================

def predict_news(news_text):
    sequence, length = encode_text(news_text)

    with torch.no_grad():
        logits = model(sequence, length)
        real_probability = torch.sigmoid(logits).item()

    fake_probability = 1.0 - real_probability

    if real_probability >= 0.5:
        prediction = "Real News"
    else:
        prediction = "Fake News"

    return prediction, fake_probability, real_probability


# ==========================================
# 6. Interactive terminal application
# ==========================================

def main():
    print("\n" + "=" * 50)
    print("       FAKE NEWS DETECTION - BiLSTM")
    print("=" * 50)

    print("Enter a news article or headline.")
    print("Type 'exit' to quit the program.")

    while True:
        print("\n" + "-" * 50)
        news_text = input("Enter news text: ").strip()

        if news_text.lower() in {"exit", "quit"}:
            print("Program closed.")
            break

        if not news_text:
            print("Please enter some news text.")
            continue

        prediction, fake_prob, real_prob = predict_news(
            news_text
        )

        print("\nPrediction Result")
        print("-" * 30)
        print(f"Predicted class: {prediction}")
        print(f"Fake probability: {fake_prob * 100:.2f}%")
        print(f"Real probability: {real_prob * 100:.2f}%")
        print("-" * 30)
        print(
            "Note: These are model probabilities, "
            "not proof that the news is true or false."
        )


if __name__ == "__main__":
    main()
