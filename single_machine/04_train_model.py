import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("/mnt/newdisk/anass/final_dataset.csv")

y = df["failed"]
X = df.drop(columns=["failed","sid","day"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    class_weight="balanced"
)

model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:,1]
pred  = (probs > 0.9).astype(int)

print("Precision:", precision_score(y_test, pred))
print("Recall:", recall_score(y_test, pred))
print("F1:", f1_score(y_test, pred))
