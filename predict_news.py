
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "models" / "tfidf_logistic_regression.joblib"

model = joblib.load(model_path)

print("Fake News Detection System")
print("Type 'exit' to stop.\n")

while True:
    news = input("Enter news text: ").strip()

    if news.lower() == "exit":
        break

    if not news:
        print("Please enter some news text.\n")
        continue

    prediction = model.predict([news])[0]
    probabilities = model.predict_proba([news])[0]

    confidence = max(probabilities) * 100

    if prediction == 0:
        result = "Fake News"
    else:
        result = "Real News"

    print("Prediction:", result)
    print(f"Model confidence: {confidence:.2f}%\n")
