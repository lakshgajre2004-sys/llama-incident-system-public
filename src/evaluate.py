import os
import joblib
from sklearn.metrics import accuracy_score, classification_report

from src.utils import load_data

MODEL_PATH = "results/baseline_model.joblib"

print("EVALUATE SCRIPT STARTED")

def main():
    print("-> Loading data...")
    X, y = load_data()

    if not os.path.exists(MODEL_PATH):
        print("❌ Model not trained yet! Run `python -m src.train` first.")
        return

    print("-> Loading model...")
    model = joblib.load(MODEL_PATH)

    print("-> Predicting...")
    preds = model.predict(X)

    acc = accuracy_score(y, preds)
    report = classification_report(y, preds)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(report)

    os.makedirs("results", exist_ok=True)
    with open("results/evaluation.txt", "w") as f:
        f.write(f"Accuracy: {acc}\n\n{report}")

    print("\n[✔] Saved evaluation to results/evaluation.txt")

if __name__ == "__main__":
    main()
