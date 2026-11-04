from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import yfinance as yf
import pandas as pd
import os
import time

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MARKETS = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X", 
    "AUDUSD": "AUDUSD=X", "USDCAD": "CAD=X", "NZDUSD": "NZDUSD=X",
    "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD", "SOLUSD": "SOL-USD",
    "NAS100": "^NDX", "US30": "^DJI", "SPX500": "^GSPC",
    "XAUUSD": "GC=F", "XAGUSD": "SI=F", "USOIL": "CL=F"
}

class Setup(BaseModel):
    timeframe: str
    direction: str
    entry: float
    stopLoss: float
    takeProfit: float
    current_price: float
    strategy_name: str
    strategy_logic: str

class Signal(BaseModel):
    pair: str
    setups: List[Setup]

def get_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def analyze_pair(pair, symbol):
    try:
        setups = []
        
        # 1. GET H4 DATA (The Master Trend Filter)
        h4_df = yf.Ticker(symbol).history(period="3mo", interval="4h")
        if h4_df.empty or len(h4_df) < 50: return None
        
        h4_df['EMA_50'] = h4_df['Close'].ewm(span=50, adjust=False).mean()
        h4_df['EMA_200'] = h4_df['Close'].ewm(span=200, adjust=False).mean()
        last_h4 = h4_df.iloc[-2]
        
        if last_h4['EMA_50'] > last_h4['EMA_200']:
            master_trend = "UP"
        elif last_h4['EMA_50'] < last_h4['EMA_200']:
            master_trend = "DOWN"
        else:
            return None # Market is ranging

        # 2. SCAN TIMEFRAMES (M15, M30, H1, H4)
        timeframes_config = [
            {"tf": "15m", "period": "5d",  "type": "Scalping",   "sl_mult": 1.0,  "tp_mult": 2.0},  # 1:2 RR
            {"tf": "30m", "period": "10d", "type": "Day Trading","sl_mult": 1.2,  "tp_mult": 3.6},  # 1:3 RR (The M30 Bridge)
            {"tf": "1h",  "period": "1mo", "type": "Intraday",   "sl_mult": 1.5,  "tp_mult": 3.75}, # 1:2.5 RR
            {"tf": "4h",  "period": "3mo", "type": "Swing",      "sl_mult": 2.0,  "tp_mult": 6.0}   # 1:3 RR
        ]

        for config in timeframes_config:
            tf = config["tf"]
            df = yf.Ticker(symbol).history(period=config["period"], interval=tf)
            
            if df.empty or len(df) < 30: continue
            
            df['ATR'] = get_atr(df)
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            
            last = df.iloc[-2] 
            current_price = df.iloc[-1]['Close'] 
            atr = last['ATR']
            decimals = 5 if current_price < 10 else 2
            
            sl_dist = atr * config["sl_mult"]
            tp_dist = atr * config["tp_mult"]
            
            prev_high = df['High'].iloc[-5:-1].max()
            prev_low = df['Low'].iloc[-5:-1].min()
            
            # BUY LOGIC
            if master_trend == "UP":
                broke_structure = last['High'] > prev_high
                pulled_back = last['Low'] <= last['EMA_20'] * 1.002
                rejected = last['Close'] > last['EMA_20']
                
                if broke_structure and pulled_back and rejected:
                    entry = round(last['Close'], decimals)
                    if abs(current_price - entry) <= sl_dist:
                        setups.append(Setup(
                            timeframe=tf.upper(), direction="BUY", entry=entry,
                            stopLoss=round(entry - sl_dist, decimals),
                            takeProfit=round(entry + tp_dist, decimals),
                            current_price=round(current_price, decimals),
                            strategy_name=f"{config['type']}: BOS Pullback",
                            strategy_logic=f"H4 Trend UP. {tf.upper()} broke structure & pulled back to EMA 20."
                        ))

            # SELL LOGIC
            elif master_trend == "DOWN":
                broke_structure = last['Low'] < prev_low
                pulled_back = last['High'] >= last['EMA_20'] * 0.998
                rejected = last['Close'] < last['EMA_20']
                
                if broke_structure and pulled_back and rejected:
                    entry = round(last['Close'], decimals)
                    if abs(current_price - entry) <= sl_dist:
                        setups.append(Setup(
                            timeframe=tf.upper(), direction="SELL", entry=entry,
                            stopLoss=round(entry + sl_dist, decimals),
                            takeProfit=round(entry - tp_dist, decimals),
                            current_price=round(current_price, decimals),
                            strategy_name=f"{config['type']}: BOS Pullback",
                            strategy_logic=f"H4 Trend DOWN. {tf.upper()} broke structure & pulled back to EMA 20."
                        ))

        if setups:
            return Signal(pair=pair, setups=setups)
        return None

    except Exception as e:
        return None

@app.get("/signals")
def get_signals():
    final_signals = []
    for pair, symbol in MARKETS.items():
        signal = analyze_pair(pair, symbol)
        if signal:
            final_signals.append(signal)
        time.sleep(0.2) 
    return final_signals

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
