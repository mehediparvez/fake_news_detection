
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load dataset
file_path = BASE_DIR / "data" / "WELFake_Dataset.csv"
df = pd.read_csv(file_path)

print("Original shape:", df.shape)

# Remove unnecessary index column
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Rows need article text; a missing title is treated as empty
df = df.dropna(subset=["text"]).copy()
df["title"] = df["title"].fillna("").astype(str).str.strip()
df["text"] = df["text"].astype(str).str.strip()

# Remove empty text rows
df = df[df["text"].ne("")].copy()

# Remove the "CITY (Reuters) - " dateline. Almost every real article
# starts with it, so models would learn the source instead of content.
df["text"] = df["text"].str.replace(
    r"^[^()]{0,100}?\(Reuters\)\s*-\s*",
    "",
    regex=True,
)

# Combine title and article text
df["content"] = (df["title"] + " " + df["text"]).str.strip()

# Remove duplicate articles so the same text cannot appear
# in both the training and test sets
df = df.drop_duplicates(subset="content").copy()

# WELFake's documented mapping (0 = fake, 1 = real) is inverted in the
# actual data: Reuters articles are labelled 0 and known fake sites 1.
# Flip it so the rest of the project can use 0 = Fake, 1 = Real.
df["label"] = 1 - df["label"].astype(int)

# Keep only required columns
df = df[["content", "label"]]

# Save cleaned dataset
output_path = BASE_DIR / "data" / "WELFake_cleaned.csv"
df.to_csv(output_path, index=False)

print("\nCleaning completed!")
print("Cleaned shape:", df.shape)

print("\nLabel distribution (0 = Fake, 1 = Real):")
print(df["label"].value_counts().sort_index())

print("\nMissing values:")
print(df.isnull().sum())

print("\nSaved to:", output_path)
