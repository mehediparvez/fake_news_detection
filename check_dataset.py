
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "data" / "WELFake_Dataset.csv"

print("Checking dataset path...")

if not file_path.exists():
    print("CSV file not found!")
    print("Expected location:", file_path.resolve())
else:
    df = pd.read_csv(file_path)

    print("\nDataset loaded successfully!")
    print("Dataset shape:", df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())
