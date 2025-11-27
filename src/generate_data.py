import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import os

os.makedirs("data", exist_ok=True)

def generate_classification(n_samples=2000, n_features=20, n_informative=10, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        random_state=random_state
    )

    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    return df

if __name__ == "__main__":
    df = generate_classification()
    out_path = os.path.join("data", "dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved dataset to {out_path} (shape={df.shape})")
