# src/train.py  (instrumented debug version)
import os
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.model import build_baseline_model
from src.utils import load_data

os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

MODEL_PATH = os.path.join("results", "baseline_model.joblib")

print("TRAIN SCRIPT STARTED")
start_all = time.time()

print("-> Loading data...")
t0 = time.time()
X, y = load_data()
print(f"   loaded data in {time.time()-t0:.3f}s; X.shape={X.shape}")

print("-> Splitting data...")
t0 = time.time()
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   split in {time.time()-t0:.3f}s; X_train.shape={X_train.shape}")

print("-> Building model...")
t0 = time.time()
model = build_baseline_model()
print(f"   model built in {time.time()-t0:.3f}s")

print("-> Starting fit() (this is where heavy work happens)...")
t0 = time.time()
try:
    model.fit(X_train, y_train)
except Exception as e:
    print("!!! Exception during fit():", e)
    raise
print(f"   fit finished in {time.time()-t0:.3f}s")

print("-> Predicting on validation set...")
t0 = time.time()
preds = model.predict(X_val)
acc = accuracy_score(y_val, preds)
print(f"   predict+metric in {time.time()-t0:.3f}s; accuracy={acc:.4f}")

joblib.dump(model, MODEL_PATH)
with open("logs/train_log.txt", "w") as f:
    f.write(f"Validation Accuracy: {acc}\n")

print(f"[✔] Model trained. Accuracy = {acc:.4f}")
print(f"[✔] Saved model to {MODEL_PATH}")
print("TOTAL TIME:", time.time() - start_all)
