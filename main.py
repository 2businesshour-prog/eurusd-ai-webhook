from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import os

app = FastAPI()

DATA_FILE = "eurusd_1m.csv"

@app.get("/")
def home():
    return {"status": "EURUSD Binary AI System Running"}

@app.get("/run-system")
def run_system():

    if not os.path.exists(DATA_FILE):
        return {"error": "Upload eurusd_1m.csv to repository"}

    # === LOAD MT5 DATA ===
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

    # ATR
    df["atr"] = (df["high"] - df["low"]).rolling(7).mean()
    df["atr_pct"] = df["atr"] / df["atr"].rolling(200).mean()

    # Trend strength proxy
    df["trend_strength"] = abs(df["ema_spread"])

    # Candle body %
    df["body"] = abs(df["close"] - df["open"])
    df["range"] = df["high"] - df["low"]
    df["body_pct"] = df["body"] / df["range"]

    # Timezone conversion UTC+2 → IST
    df["time_utc"] = df["time"] - pd.Timedelta(hours=2)
    df["time_ist"] = df["time_utc"].dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
    df["hour"] = df["time_ist"].dt.hour

    # Target (next candle)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    df.dropna(inplace=True)

    # === REGIME CLUSTERING ===
    regime_features = df[["atr_pct","trend_strength","body_pct"]]

    kmeans = KMeans(n_clusters=4, random_state=42)
    df["regime"] = kmeans.fit_predict(regime_features)

    # === TRAIN MODEL PER REGIME ===
    results = {}

    total_trades = 0
    total_wins = 0

    for regime in df["regime"].unique():

        sub = df[df["regime"] == regime]

        if len(sub) < 500:
            continue

        features = sub[[
            "ema_spread","rsi","atr","atr_pct",
            "trend_strength","body_pct","hour"
        ]]

        target = sub["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, shuffle=False
        )

        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:,1]

        sub_test = sub.iloc[-len(X_test):].copy()
        sub_test["pred"] = preds
        sub_test["prob"] = probs

        sub_test["win"] = 0
        sub_test.loc[
            ((sub_test["pred"] == 1) & (sub_test["target"] == 1)) |
            ((sub_test["pred"] == 0) & (sub_test["target"] == 0)),
            "win"
        ] = 1

        regime_trades = len(sub_test)
        regime_winrate = sub_test["win"].mean()

        high_conf = sub_test[sub_test["prob"] > 0.72]

        if len(high_conf) > 0:
            high_conf_winrate = high_conf["win"].mean()
        else:
            high_conf_winrate = 0

        results[f"regime_{regime}"] = {
            "trades": regime_trades,
            "winrate": float(regime_winrate),
            "high_conf_trades": len(high_conf),
            "high_conf_winrate": float(high_conf_winrate)
        }

        total_trades += regime_trades
        total_wins += sub_test["win"].sum()

    overall_winrate = total_wins / total_trades if total_trades > 0 else 0

    return {
        "overall_trades": total_trades,
        "overall_winrate": float(overall_winrate),
        "regime_breakdown": results
    }
