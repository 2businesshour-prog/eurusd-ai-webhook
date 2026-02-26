from fastapi import FastAPI, Request
import pandas as pd
import os
from datetime import datetime
import joblib
import uvicorn

app = FastAPI()

MODEL_PATH = "model.pkl"
LOG_FILE = "signals_log.csv"

# Load model if exists
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

@app.get("/")
def home():
    return {"status": "EURUSD AI Webhook Running"}

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()

    now = datetime.utcnow().isoformat()

    signal = data.get("signal")
    price = float(data.get("price"))
    ema9 = float(data.get("ema9"))
    ema21 = float(data.get("ema21"))
    rsi = float(data.get("rsi"))
    atr = float(data.get("atr"))
    hour = int(data.get("hour"))

    ema_spread = ema9 - ema21

    features = [[ema_spread, rsi, atr, hour]]

    probability = None

    if model:
        probability = float(model.predict_proba(features)[0][1])

    log_data = {
        "time": now,
        "signal": signal,
        "price": price,
        "ema_spread": ema_spread,
        "rsi": rsi,
        "atr": atr,
        "hour": hour,
        "probability": probability
    }

    df = pd.DataFrame([log_data])

    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)

    return {"status": "received", "probability": probability}
