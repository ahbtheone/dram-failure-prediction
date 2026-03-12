import pandas as pd
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score

# ================= CONFIG =================
DATA_PATH = "/mnt/newdisk/anass/final_dataset.csv"
WINDOW_K = 7
WARMUP_DAYS = 30

FEATURES = [
    "ce_count",
    "unique_banks",
    "unique_rows",
    "mean_inter_error_time"
]

# ================= LOAD =================
print("Loading daily data...")
df = pd.read_csv(DATA_PATH, parse_dates=["day"])
df = df.sort_values(["sid", "day"])

print("Total rows:", len(df))
print("Failures:", df["failed"].sum())

# ================= ONLINE LOOP =================
y_true, y_pred = [], []
train_X, train_y = [], []

model = None
day_counter = 0

for sid, g in df.groupby("sid"):
    g = g.reset_index(drop=True)

    for i in range(len(g)):
        current_day = g.loc[i, "day"]

        # sliding window
        window = g.iloc[max(0, i - WINDOW_K):i]

        if window.empty:
            continue

        # aggregate window features
        feat = {
            "ce_count": window["ce_count_past_x"].sum(),
            "unique_banks": window["unique_banks_past_x"].max(),
            "unique_rows": window["unique_rows_past_x"].max(),
            "mean_inter_error_time": window["mean_inter_error_time_past_x"].mean(),
        }

        label = g.loc[i, "failed"]

        day_counter += 1

        # warm-up phase
        if day_counter <= WARMUP_DAYS:
            train_X.append(list(feat.values()))
            train_y.append(label)
            continue

        # initialize model
        if model is None:
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                class_weight="balanced",
                random_state=42
            )
            model.fit(train_X, train_y)
            print("Model initialized at step", day_counter)

        # predict
        pred = model.predict([list(feat.values())])[0]

        y_true.append(label)
        y_pred.append(pred)

        # online update
        train_X.append(list(feat.values()))
        train_y.append(label)

        model.fit(train_X, train_y)

# ================= METRICS =================
print("\nONLINE SLIDING-WINDOW EVALUATION")
print("Samples evaluated:", len(y_true))

if len(y_true) > 0:
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1:", f1_score(y_true, y_pred, zero_division=0))
else:
    print("No evaluable samples.")
