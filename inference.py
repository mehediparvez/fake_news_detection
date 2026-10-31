from pathlib import Path
import json
import re

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


# ==========================================
# 1. Project configuration
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "bilstm" / "bilstm_model.pth"
VOCAB_PATH = BASE_DIR / "models" / "bilstm" / "vocab.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. BiLSTM model architecture
# Must match train_bilstm.py
# ==========================================

class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=64,
        dropout=0.3
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, lengths):
        embedded = self.embedding(x)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined = torch.cat(
            (forward_hidden, backward_hidden), dim=1
        )

        combined = self.dropout(combined)

        return self.fc(combined).squeeze(1)


# ==========================================
# 3. Load model and vocabulary
# ==========================================

def load_model():  # callers should cache the result
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
        weights_only=True
    )

    model = BiLSTMClassifier(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    max_length = checkpoint["max_length"]

    return model, vocab, max_length


# ==========================================
# 4. Prepare news text
# ==========================================

def prepare_text(text, vocab, max_length):
    tokens = re.findall(r"\b\w+\b", str(text).lower())

    if not tokens:
        return None, None

    token_ids = [
        vocab.get(token, vocab.get("<UNK>", 1))
        for token in tokens[:max_length]
    ]

    original_length = len(token_ids)

    token_ids += [0] * (max_length - original_length)

    input_tensor = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE
    )

    length_tensor = torch.tensor(
        [original_length],
        dtype=torch.long
    )

    return input_tensor, length_tensor


# ==========================================
# 5. Predict news
# ==========================================

def predict_news(text, model, vocab, max_length):
    input_tensor, length_tensor = prepare_text(
        text, vocab, max_length
    )

    if input_tensor is None:
        return None

    with torch.inference_mode():
        logits = model(input_tensor, length_tensor)
        real_probability = torch.sigmoid(logits).item()

    fake_probability = 1.0 - real_probability

    if real_probability >= 0.5:
        prediction = "Real News"
    else:
        prediction = "Fake News"

    return {
        "prediction": prediction,
        "fake_probability": fake_probability,
        "real_probability": real_probability
    }
