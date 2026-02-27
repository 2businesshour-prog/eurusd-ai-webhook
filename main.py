from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import os
import itertools

app = FastAPI()
DATA_FILE = "eurusd_1m.csv"

PAYOUT = 0.85

@app.get("/")
def home():
    return {"status": "EURUSD Binary Research Engine Running"}

@app.get("/run-system")
def run_system():

    if not os.path.exists(DATA_FILE):
        return {"error": "Upload eurusd_1m.csv"}

    df = pd.read_csv(DATA_FILE, sep="\t")
    df["time"] = pd.to_datetime(df["<DATE>"] + " " + df["<TIME>"])
    df = df.sort_values("time")

    df["open"] = df["<OPEN>"]
    df["high"] = df["<HIGH>"]
    df["low"] = df["<LOW>"]
    df["close"] = df["<CLOSE>"]

    df = df[["time","open","high","low","close"]]

    # RESAMPLE TO 5m
    df_5m = df.set_index("time").resample("5min").agg({
        "open":"first",
        "high":"max",
        "low":"min",
        "close":"last"
    }).dropna().reset_index()

    if len(df_5m) < 2000:
        return {"error": "Not enough 5m data"}

    # TARGETS
    df_5m["target_10m"] = (df_5m["close"].shift(-2) > df_5m["close"]).astype(int)
    df_5m["target_15m"] = (df_5m["close"].shift(-3) > df_5m["close"]).astype(int)
    df_5m.dropna(inplace=True)

    results = []

    # PARAMETER GRID
    ema_short_list = [10, 20, 30]
    ema_long_list = [40, 60, 80]
    rsi_lengths = [7, 14, 21]
    atr_lengths = [7, 14, 21]
    prob_thresholds = [0.55, 0.60, 0.65]

    param_grid = list(itertools.product(
        ema_short_list,
        ema_long_list,
        rsi_lengths,
        atr_lengths,
        prob_thresholds
    ))

    for ema_s, ema_l, rsi_len, atr_len, prob_th in param_grid:

        if ema_s >= ema_l:
            continue

        df_tmp = df_5m.copy()

        # FEATURES
        df_tmp["ema_s"] = df_tmp["close"].ewm(span=ema_s).mean()
        df_tmp["ema_l"] = df_tmp["close"].ewm(span=ema_l).mean()
        df_tmp["ema_spread"] = df_tmp["ema_s"] - df_tmp["ema_l"]

        delta = df_tmp["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(rsi_len).mean()
        avg_loss = loss.rolling(rsi_len).mean()

        rs = avg_gain / (avg_loss + 1e-9)
        df_tmp["rsi"] = 100 - (100 / (1 + rs))

        df_tmp["range"] = df_tmp["high"] - df_tmp["low"]
        df_tmp["atr"] = df_tmp["range"].rolling(atr_len).mean()

        df_tmp.dropna(inplace=True)

        features = ["ema_spread", "rsi", "atr"]

        # WALK FORWARD SPLITS
        n = len(df_tmp)
        split1 = int(n * 0.70)
        split2 = int(n * 0.85)

        folds = [
            (0, split1, split1, split2),
            (0, split2, split2, n)
        ]

        ev_scores = []

        for train_start, train_end, test_start, test_end in folds:

            train = df_tmp.iloc[train_start:train_end]
            test = df_tmp.iloc[test_start:test_end]

            if len(test) < 200:
                continue

            X_train = train[features]
            y_train = train["target_15m"]

            X_test = test[features]
            y_test = test["target_15m"]

            model = GradientBoostingClassifier()
            model.fit(X_train, y_train)

            probs = model.predict_proba(X_test)[:,1]
            preds = (probs > prob_th).astype(int)

            wins = ((preds == 1) & (y_test == 1)) | ((preds == 0) & (y_test == 0))
            winrate = wins.mean()

            ev = (winrate * PAYOUT) - ((1 - winrate) * 1)
            ev_scores.append(ev)

        if len(ev_scores) == 2:
            avg_ev = np.mean(ev_scores)
            stability = abs(ev_scores[0] - ev_scores[1])

            results.append({
                "ema_short": ema_s,
                "ema_long": ema_l,
                "rsi_len": rsi_len,
                "atr_len": atr_len,
                "prob_threshold": prob_th,
                "avg_ev": float(avg_ev),
                "stability": float(stability)
            })

    if not results:
        return {"error": "No valid configurations"}

    df_results = pd.DataFrame(results)
    df_results = df_results[df_results["avg_ev"] > 0]
    df_results = df_results.sort_values(
        by=["avg_ev", "stability"],
        ascending=[False, True]
    )

    top_configs = df_results.head(10)

    return top_configs.to_dict(orient="records")
