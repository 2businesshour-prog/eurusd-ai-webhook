from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import os

app = FastAPI()
DATA_FILE = "eurusd_1m.csv"

@app.get("/")
def home():
    return {"status": "EURUSD Binary AI V3 Running"}

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

    # BASIC FEATURES
    df["range"] = df["high"] - df["low"]
    df["atr"] = df["range"].rolling(7).mean()
    df["atr_pct"] = df["atr"] / df["atr"].rolling(200).mean()

    df["avg_range5"] = df["range"].rolling(5).mean()
    df["compression"] = (df["atr_pct"] < 0.7).astype(int)

    df["acceleration"] = df["range"] / df["avg_range5"]

    # Breakout detection
    df["recent_high"] = df["high"].rolling(5).max().shift(1)
    df["recent_low"] = df["low"].rolling(5).min().shift(1)

    df["break_up"] = (df["high"] > df["recent_high"]).astype(int)
    df["break_down"] = (df["low"] < df["recent_low"]).astype(int)

    # Expansion trigger
    df["expansion"] = (df["acceleration"] > 1.5).astype(int)

    # Edge Zone definition
    df["edge_zone"] = (
        (df["compression"] == 1) &
        (df["expansion"] == 1) &
        ((df["break_up"] == 1) | (df["break_down"] == 1))
    ).astype(int)

    # Direction of breakout candle
    df["direction"] = np.where(df["break_up"] == 1, 1,
                        np.where(df["break_down"] == 1, 0, np.nan))

    # Target = next candle direction
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    df.dropna(inplace=True)

    eligible = df[df["edge_zone"] == 1].copy()

    if len(eligible) < 300:
        return {"error": "Not enough edge trades detected"}

    # MODEL FEATURES
    features = [
        "atr_pct",
        "acceleration",
        "range"
    ]

    X = eligible[features]
    y = eligible["target"]

    split = int(len(X)*0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:,1]
    preds = model.predict(X_test)

    eligible_test = eligible.iloc[-len(X_test):].copy()
    eligible_test["pred"] = preds
    eligible_test["prob"] = probs

    eligible_test["win"] = (
        ((eligible_test["pred"] == 1) & (eligible_test["target"] == 1)) |
        ((eligible_test["pred"] == 0) & (eligible_test["target"] == 0))
    ).astype(int)

    raw_winrate = eligible_test["win"].mean()

    high_conf = eligible_test[eligible_test["prob"] > 0.74]

    filtered_winrate = high_conf["win"].mean() if len(high_conf) > 0 else 0

    return {
        "eligible_trades": len(eligible_test),
        "raw_winrate": float(raw_winrate),
        "high_conf_trades": len(high_conf),
        "filtered_winrate": float(filtered_winrate)
    }
