# src/publish_pipeline.py
"""
Publication-ready evaluation pipeline.

Usage:
    python -m src.publish_pipeline

Outputs (in results/):
 - final_model.joblib          # trained on train+val with best hyperparams
 - publish_evaluation.txt      # test set metrics & classification report
 - publish_metrics.json        # numeric metrics (accuracy, auc, etc.)
 - confusion_matrix.png
 - roc_curve.png (if probabilities available)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from time import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Paths
DATA_PATH = "data/dataset.csv"
RESULTS_DIR = "results"
FINAL_MODEL_PATH = os.path.join(RESULTS_DIR, "final_model.joblib")
EVAL_TEXT_PATH = os.path.join(RESULTS_DIR, "publish_evaluation.txt")
EVAL_JSON_PATH = os.path.join(RESULTS_DIR, "publish_metrics.json")
CM_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")
ROC_PATH = os.path.join(RESULTS_DIR, "roc_curve.png")

os.makedirs(RESULTS_DIR, exist_ok=True)

RND = 42

def build_pipeline(solver="liblinear", C=1.0, max_iter=1000):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver=solver, C=C, max_iter=max_iter, random_state=RND))
    ])

def main():
    t0 = time()
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")

    # 1) Load data
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"])
    y = df["target"]

    # 2) Create reproducible splits: train, val, test (60/20/20)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RND)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RND)
    # Note: 0.25 of remaining 0.8 => 0.2 of total -> so train:0.6 val:0.2 test:0.2

    # 3) Grid search on training set (use cross-validation inside GridSearch)
    param_grid = [
        {"clf__solver": ["liblinear"], "clf__C": [0.01, 0.1, 1, 10, 100]},
        {"clf__solver": ["saga"], "clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l2"], "clf__max_iter": [1000]},
    ]
    base_pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=RND))])
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RND)
    gs = GridSearchCV(
        estimator=base_pipe,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv_outer,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    print("[1/5] Performing GridSearchCV on training set (with internal 5-fold CV). This may take some time...")
    gs_t0 = time()
    gs.fit(X_train, y_train)
    print(f"GridSearchCV finished in {time()-gs_t0:.2f}s. Best score (cv): {gs.best_score_:.4f}")
    print("Best params:", gs.best_params_)

    # 4) Evaluate best estimator on validation (optional) and then retrain on train+val
    best = gs.best_estimator_

    # Validate (informative, not for publication)
    val_preds = best.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"Validation accuracy (informative): {val_acc:.4f}")

    # Retrain on train+val for final model
    X_final_train = pd.concat([X_train, X_val], axis=0)
    y_final_train = pd.concat([y_train, y_val], axis=0)

    print("[2/5] Retraining final model on train+val with best hyperparameters...")
    final_t0 = time()
    best.fit(X_final_train, y_final_train)
    joblib.dump(best, FINAL_MODEL_PATH)
    print(f"Final model saved to {FINAL_MODEL_PATH} (trained in {time()-final_t0:.2f}s)")

    # 5) Evaluate on held-out test set (UNBIASED estimate!)
    print("[3/5] Evaluating final model on held-out TEST set...")
    preds_test = best.predict(X_test)
    acc_test = accuracy_score(y_test, preds_test)
    clf_report = classification_report(y_test, preds_test)
    cm = confusion_matrix(y_test, preds_test)

    # Save textual evaluation
    with open(EVAL_TEXT_PATH, "w", encoding="utf-8") as f:
        f.write("UNBIASED EVALUATION (TEST SET)\n")
        f.write(f"Accuracy: {acc_test:.6f}\n\n")
        f.write("Classification report:\n")
        f.write(clf_report)
        f.write("\nConfusion matrix:\n")
        f.write(np.array2string(cm))
    # Save numeric metrics
    metrics = {
        "accuracy": float(acc_test),
        "n_test": int(len(y_test)),
        "time_seconds": time() - t0,
        "best_cv_score": float(gs.best_score_),
        "best_params": gs.best_params_
    }
    # Try ROC/AUC (binary only)
    roc_auc_val = None
    if len(np.unique(y_test)) == 2:
        probas = None
        try:
            probas = best.predict_proba(X_test)[:, 1]
        except Exception:
            try:
                probas = best.decision_function(X_test)
                probas = (probas - probas.min()) / (probas.max() - probas.min())
            except Exception:
                probas = None
        if probas is not None:
            fpr, tpr, _ = roc_curve(y_test, probas)
            roc_auc_val = float(auc(fpr, tpr))
            # save ROC plot
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc_val:.3f}")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve (Test set)")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(ROC_PATH)
            plt.close()

    # Save confusion matrix image
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Test set)")
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(CM_PATH)
    plt.close()

    metrics["roc_auc"] = roc_auc_val
    with open(EVAL_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[4/5] Saved evaluation artifacts:")
    print(" -", EVAL_TEXT_PATH)
    print(" -", EVAL_JSON_PATH)
    print(" -", CM_PATH)
    if roc_auc_val is not None:
        print(" -", ROC_PATH)

    print("\n[✔] Unbiased evaluation pipeline finished in:", round(time() - t0, 2), "s")
    print(f"Test accuracy (unbiased): {acc_test:.4f}")
    if roc_auc_val is not None:
        print(f"Test AUC: {roc_auc_val:.4f}")

if __name__ == "__main__":
    main()
