import joblib
import pandas as pd
from sklearn.metrics import classification_report

model = joblib.load('../results/sample_model.joblib')

df = pd.read_csv('../datasets/sample_dataset_1000.csv')
X = df.drop(columns=['target'])
y = df['target']

preds = model.predict(X)

print("Evaluation Report:\n")
print(classification_report(y, preds))
