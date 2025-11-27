import pandas as pd, os

os.makedirs("datasets", exist_ok=True)

df = pd.read_csv("../data/dataset.csv")
df.sample(1000, random_state=42).to_csv("datasets/sample_dataset_1000.csv", index=False)

print("Saved datasets/sample_dataset_1000.csv")
