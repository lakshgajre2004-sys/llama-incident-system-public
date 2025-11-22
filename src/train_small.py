import joblib, os, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

os.makedirs('../results', exist_ok=True)

df = pd.read_csv('../datasets/sample_dataset_1000.csv')
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=500, solver='liblinear'))
])

pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)
acc = accuracy_score(y_test, preds)

joblib.dump(pipe, '../results/sample_model.joblib')

with open('../results/sample_eval.txt', 'w') as f:
    f.write(f"accuracy={acc}\n")

print("Training complete. Accuracy:", acc)
