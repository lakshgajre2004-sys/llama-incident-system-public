# src/utils.py
import pandas as pd
import numpy as np

def load_data(path="data/dataset.csv"):
    """
    Loads dataset.csv, drops the target column and any non-numeric columns
    (e.g., 'hour' datetime strings), and returns X (numeric) and y.
    """
    df = pd.read_csv(path)

    # Ensure 'target' exists
    if 'target' not in df.columns:
        raise KeyError("Expected 'target' column in dataset.csv")

    # Separate target
    y = df['target']

    # Drop target and keep numeric columns only
    X = df.drop(columns=['target'])

    # If 'hour' or other datetime-like columns are present, drop them
    # Drop columns with non-numeric dtype
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        # Log dropped columns for visibility
        print(f"[utils.load_data] Dropping non-numeric columns from X: {non_numeric}")
        X = X.drop(columns=non_numeric)

    # Optional: replace inf / NaN with numeric safe values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X, y
