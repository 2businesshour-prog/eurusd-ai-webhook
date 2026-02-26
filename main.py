from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import os

app = FastAPI()
DATA_FILE = "eurusd_1m.csv"

@app.get("/")
def home():
    return {"status": "EURUSD Binary AI V4 Multi-Horizon Running"}

@app.get("/run-system")
def run_system():

    if not os.path.exists(DATA_FILE):
        return {"error": "Upload eurusd_1m.csv"}

    # LOAD DATA
    df = pd.read_csv(DATA_FILE, sep="\t")
    df["time"] = pd.to_datetime(df["<DATE>"] + " " + df["<TIME>"])
    df = df.sort_values("time")

    df["open"] = df["<OPEN>"]
    df["high"] = df["<HIGH>"]
    df["low"] = df["<LOW>"]
    df["close"] = df["<CLOSE>"]

    df = df[["time","open","high","low","close"]]

    # FEATURES
    df["range"] = df["high"] - df["low"]
    df["atr"] = df["range"].rolling(7).mean()
    df["atr_pct"] = df["atr"] / df["atr"].rolling(200).mean()

    df["avg_range5"] = df["range"].rolling(5).mean()
    df["acceleration"] = df["range"] / df["avg_range5"]

    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema21"] = df["close"].ewm(span=21).mean()
    df["ema_spread"] = df["ema9"] - df["ema21"]

    df["rsi"] = 100 - (100 / (1 + 
        (df["close"].diff().clip(lower=0).rolling(7).mean() /
         (-df["close"].diff().clip(upper=0).rolling(7).mean()))
    ))

    df["hour"] = df["time"].dt.hour
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

    df.dropna(inplace=True)

    # STRUCTURAL FILTER (compression → expansion)
    df["compression"] = df["atr_pct"] < 0.8
    df["expansion"] = df["acceleration"] > 1.3

    df["edge_zone"] = (df["compression"] & df["expansion"]).astype(int)

    eligible = df[df["edge_zone"] == 1].copy()

    if len(eligible) < 500:
        return {"error": "Not enough structural setups"}

    # MULTI-HORIZON TARGETS
    horizons = [1,2,3,4,5]
    results = {}

    for h in horizons:
        eligible[f"target_{h}"] = (
            eligible["close"].shift(-h) > eligible["close"]
        ).astype(int)

    eligible.dropna(inplace=True)

    features = [
        "atr_pct",
        "acceleration",
        "ema_spread",
        "rsi",
        "hour_sin",
        "hour_cos"
    ]

    split = int(len(eligible)*0.8)

    summary = {}

    for h in horizons:

        X = eligible[features]
        y = eligible[f"target_{h}"]

        X_train = X.iloc[:split]
        X_test = X.iloc[split:]
        y_train = y.iloc[:split]
        y_test = y.iloc[split:]

        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:,1]
        preds = model.predict(X_test)

        test_df = eligible.iloc[split:].copy()
        test_df["pred"] = preds
        test_df["prob"] = probs

        test_df["win"] = (
            ((test_df["pred"] == 1) & (test_df[f"target_{h}"] == 1)) |
            ((test_df["pred"] == 0) & (test_df[f"target_{h}"] == 0))
        ).astype(int)

        raw_winrate = test_df["win"].mean()

        high_conf = test_df[test_df["prob"] > 0.60]

        filtered_winrate = high_conf["win"].mean() if len(high_conf) > 0 else 0

        summary[f"{h}_minute"] = {
            "trades": len(test_df),
            "raw_winrate": float(raw_winrate),
            "high_conf_trades": len(high_conf),
            "filtered_winrate": float(filtered_winrate)
        }

    return summary
