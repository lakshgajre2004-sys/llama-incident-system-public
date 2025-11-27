# src/visualize.py
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from src.utils import load_data
import numpy as np
import os

os.makedirs("results", exist_ok=True)

model = joblib.load("results/baseline_model.joblib")
X, y = load_data()
preds = model.predict(X)
probas = None
try:
    probas = model.predict_proba(X)[:, 1]
except Exception:
    # fallback: try decision_function
    try:
        probas = model.decision_function(X)
        probas = (probas - probas.min()) / (probas.max() - probas.min())
    except Exception:
        probas = None

# Confusion matrix
cm = confusion_matrix(y, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")
plt.close()

# ROC curve (if probabilities available)
if probas is not None:
    fpr, tpr, _ = roc_curve(y, probas)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig("results/roc_curve.png")
    plt.close()
else:
    print("No probability scores available to plot ROC.")

print("Saved: results/confusion_matrix.png" + (", results/roc_curve.png" if probas is not None else ""))
