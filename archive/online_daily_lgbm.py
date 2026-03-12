import pandas as pd
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score

# ===================== CONFIG =====================
DATA_PATH = "/mnt/newdisk/anass/daily_scaledfp_subset.csv"
WARMUP_DAYS = 30      # first week as warm-up
MIN_POS = 1          # minimum positives to train

FEATURES = [
    "ce_count_past",
    "unique_banks_past",
    "unique_rows_past",
    "mean_inter_error_time_past"
]

# ===================== LOAD DATA =====================
print("Loading daily features...")
df = pd.read_csv(DATA_PATH)
df["day"] = pd.to_datetime(df["day"])

df = df.sort_values(["sid", "day"])

print("Total rows:", len(df))
print("Failures:", df["failed"].sum())

# ===================== ONLINE LOOP =====================
y_true = []
y_pred = []

for sid, df_sid in df.groupby("sid"):
    df_sid = df_sid.sort_values("day").reset_index(drop=True)

    if len(df_sid) <= WARMUP_DAYS:
        continue

    history = df_sid.iloc[:WARMUP_DAYS].copy()

    for i in range(WARMUP_DAYS, len(df_sid)):
        train_df = history

        # skip if no signal
        if train_df["failed"].sum() < MIN_POS:
            history = pd.concat([history, df_sid.iloc[[i]]])
            continue

        X_train = train_df[FEATURES]
        y_train = train_df["failed"]

        model = lgb.LGBMClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=4,
            class_weight="balanced",
            random_state=42
        )

        model.fit(X_train, y_train)

        test_row = df_sid.iloc[[i]]
        X_test = test_row[FEATURES]
        y_test = test_row["failed"].values[0]

        pred = model.predict(X_test)[0]

        y_true.append(y_test)
        y_pred.append(pred)

        history = pd.concat([history, test_row])

print("ONLINE EVALUATION")
print("Samples evaluated:", len(y_true))
print("Precision:", precision_score(y_true, y_pred, zero_division=0))
print("Recall:", recall_score(y_true, y_pred, zero_division=0))
print("F1:", f1_score(y_true, y_pred, zero_division=0))
