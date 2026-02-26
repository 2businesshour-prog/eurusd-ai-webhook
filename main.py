from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import os

app = FastAPI()
DATA_FILE = "eurusd_1m.csv"

@app.get("/")
def home():
    return {"status": "EURUSD Binary AI V5 5M Structure Running"}

@app.get("/run-system")
def run_system():

    if not os.path.exists(DATA_FILE):
        return {"error": "Upload eurusd_1m.csv"}

    # LOAD M1 DATA
    df = pd.read_csv(DATA_FILE, sep="\t")
    df["time"] = pd.to_datetime(df["<DATE>"] + " " + df["<TIME>"])
    df = df.sort_values("time")

    df["open"] = df["<OPEN>"]
    df["high"] = df["<HIGH>"]
    df["low"] = df["<LOW>"]
    df["close"] = df["<CLOSE>"]

    df = df[["time","open","high","low","close"]]

    # ===============================
    # RESAMPLE TO 5-MIN STRUCTURE
    # ===============================

    df_5m = df.set_index("time").resample("5T").agg({
        "open":"first",
        "high":"max",
        "low":"min",
        "close":"last"
    }).dropna().reset_index()

    # 5m FEATURES
    df_5m["ema20"] = df_5m["close"].ewm(span=20).mean()
    df_5m["ema50"] = df_5m["close"].ewm(span=50).mean()
    df_5m["ema_spread"] = df_5m["ema20"] - df_5m["ema50"]

    df_5m["rsi"] = 100 - (100 / (1 + 
        (df_5m["close"].diff().clip(lower=0).rolling(14).mean() /
         (-df_5m["close"].diff().clip(upper=0).rolling(14).mean()))
    ))

    df_5m["range"] = df_5m["high"] - df_5m["low"]
    df_5m["atr"] = df_5m["range"].rolling(14).mean()

    df_5m["hour"] = df_5m["time"].dt.hour
    df_5m["hour_sin"] = np.sin(2*np.pi*df_5m["hour"]/24)
    df_5m["hour_cos"] = np.cos(2*np.pi*df_5m["hour"]/24)

    df_5m.dropna(inplace=True)

    # ===============================
    # MULTI-HORIZON TARGETS
    # ===============================

    horizons = {
        "10m": 2,   # 2 x 5m bars
        "12m": 3,   # approx 15 but we align with 5m
        "15m": 3
    }

    results = {}

    for name, h in horizons.items():

        df_5m[f"target_{name}"] = (
            df_5m["close"].shift(-h) > df_5m["close"]
        ).astype(int)

    df_5m.dropna(inplace=True)

    features = [
        "ema_spread",
        "rsi",
        "atr",
        "hour_sin",
        "hour_cos"
    ]

    split = int(len(df_5m)*0.8)

    summary = {}

    for name in horizons.keys():

        X = df_5m[features]
        y = df_5m[f"target_{name}"]

        X_train = X.iloc[:split]
        X_test = X.iloc[split:]
        y_train = y.iloc[:split]
        y_test = y.iloc[split:]

        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:,1]
        preds = model.predict(X_test)

        test_df = df_5m.iloc[split:].copy()
        test_df["pred"] = preds
        test_df["prob"] = probs

        test_df["win"] = (
            ((test_df["pred"] == 1) & (test_df[f"target_{name}"] == 1)) |
            ((test_df["pred"] == 0) & (test_df[f"target_{name}"] == 0))
        ).astype(int)

        raw_winrate = test_df["win"].mean()

        high_conf = test_df[test_df["prob"] > 0.60]
        filtered_winrate = high_conf["win"].mean() if len(high_conf) > 0 else 0

        summary[name] = {
            "trades": len(test_df),
            "raw_winrate": float(raw_winrate),
            "high_conf_trades": len(high_conf),
            "filtered_winrate": float(filtered_winrate)
        }

    return summary
