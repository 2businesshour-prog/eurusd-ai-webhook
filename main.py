from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

app = FastAPI()
DATA_FILE = "eurusd_1m.csv"

@app.get("/")
def home():
    return {"status": "EURUSD Binary AI V2 Running"}

@app.get("/run-system")
def run_system():

    if not os.path.exists(DATA_FILE):
        return {"error": "Upload eurusd_1m.csv to repository"}

    # === LOAD DATA ===
    df = pd.read_csv(DATA_FILE, sep="\t")
    df["time"] = pd.to_datetime(df["<DATE>"] + " " + df["<TIME>"])
    df = df.sort_values("time")

    df["open"] = df["<OPEN>"]
    df["high"] = df["<HIGH>"]
    df["low"] = df["<LOW>"]
    df["close"] = df["<CLOSE>"]

    df = df[["time","open","high","low","close"]]

    # === FEATURE ENGINEERING ===

    # EMA
    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema21"] = df["close"].ewm(span=21).mean()
    df["ema_spread"] = df["ema9"] - df["ema21"]
    df["ema_slope"] = df["ema9"].diff()

    # RSI
    def rsi(series, period=7):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100/(1+rs))

    df["rsi"] = rsi(df["close"])
    df["rsi_slope"] = df["rsi"].diff()

    # ATR
    df["range"] = df["high"] - df["low"]
    df["atr"] = df["range"].rolling(7).mean()
    df["atr_pct"] = df["atr"] / df["atr"].rolling(200).mean()

    # Acceleration
    df["acceleration"] = df["range"] / df["range"].rolling(5).mean()

    # Breakout detection
    df["break_high"] = (df["high"] > df["high"].shift(1).rolling(3).max()).astype(int)
    df["break_low"] = (df["low"] < df["low"].shift(1).rolling(3).min()).astype(int)

    # ADX approximation
    df["trend_strength"] = abs(df["ema_spread"])

    # Consecutive candle direction
    df["direction"] = np.where(df["close"] > df["open"], 1, 0)
    df["consecutive"] = df["direction"].groupby((df["direction"] != df["direction"].shift()).cumsum()).cumcount()+1

    # Time features
    df["hour"] = df["time"].dt.hour
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

    df["day"] = df["time"].dt.dayofweek

    # Target
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    df.dropna(inplace=True)

    # === MODEL 1: EDGE DETECTOR ===
    df["edge_zone"] = (
        (df["acceleration"] > 1.2) &
        (df["atr_pct"] > 0.8) &
        (df["trend_strength"] > df["trend_strength"].median())
    ).astype(int)

    edge_features = [
        "ema_spread","ema_slope","rsi","rsi_slope",
        "atr_pct","acceleration","trend_strength",
        "hour_sin","hour_cos"
    ]

    X_edge = df[edge_features]
    y_edge = df["edge_zone"]

    split = int(len(df)*0.8)

    X_edge_train = X_edge.iloc[:split]
    X_edge_test = X_edge.iloc[split:]
    y_edge_train = y_edge.iloc[:split]
    y_edge_test = y_edge.iloc[split:]

    edge_model = GradientBoostingClassifier()
    edge_model.fit(X_edge_train, y_edge_train)

    edge_preds = edge_model.predict(X_edge_test)

    df_test = df.iloc[split:].copy()
    df_test["edge_pred"] = edge_preds

    eligible = df_test[df_test["edge_pred"] == 1]

    # === MODEL 2: DIRECTION ===
    if len(eligible) < 200:
        return {"error": "Not enough eligible trades"}

    dir_features = [
        "ema_spread","ema_slope","rsi","rsi_slope",
        "acceleration","break_high","break_low",
        "consecutive","hour_sin","hour_cos"
    ]

    X_dir = eligible[dir_features]
    y_dir = eligible["target"]

    split2 = int(len(X_dir)*0.8)

    X_dir_train = X_dir.iloc[:split2]
    X_dir_test = X_dir.iloc[split2:]
    y_dir_train = y_dir.iloc[:split2]
    y_dir_test = y_dir.iloc[split2:]

    dir_model = GradientBoostingClassifier()
    dir_model.fit(X_dir_train, y_dir_train)

    probs = dir_model.predict_proba(X_dir_test)[:,1]
    preds = dir_model.predict(X_dir_test)

    eligible_test = eligible.iloc[-len(X_dir_test):].copy()
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
