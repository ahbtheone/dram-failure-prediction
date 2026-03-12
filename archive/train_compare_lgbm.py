import pandas as pd
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# =========================
# CONFIG
# =========================
BASELINE_FILE = "/mnt/newdisk/anass/features_per_month.csv"
SCALEDFP_FILE = "/mnt/newdisk/anass/scaledfp_offline_labeled_features.csv"

TRAIN_MONTHS = ["0001-01","0001-02","0001-03","0001-04","0001-05","0001-06"]
TEST_MONTHS  = ["0001-07","0001-08"]

TARGET = "failed"
DROP_COLS = ["sid", "month"]

# =========================
# TRAIN + EVAL FUNCTION
# =========================
def train_and_eval(path, name):
    print(f"\n===== {name} =====")

    df = pd.read_csv(path)

    train = df[df["month"].isin(TRAIN_MONTHS)]
    test  = df[df["month"].isin(TEST_MONTHS)]

    X_train = train.drop(columns=DROP_COLS + [TARGET])
    y_train = train[TARGET]

    X_test = test.drop(columns=DROP_COLS + [TARGET])
    y_test = test[TARGET]

    model = lgb.LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"FAR       : {far:.4f}")

# =========================
# RUN BOTH EXPERIMENTS
# =========================
train_and_eval(BASELINE_FILE, "Baseline (Simple Features)")
train_and_eval(SCALEDFP_FILE, "ScaleDFP-style Features")
