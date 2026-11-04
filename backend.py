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

TIMEFRAMES = ["15m", "30m", "1h"]

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

def calculate_indicators(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    high_low = df['High'] - df['Low']
    tr = high_low.rolling(14).mean()
    df['ATR'] = tr
    return df

def get_sniper_setups(pair, symbol, tf):
    try:
        df = yf.Ticker(symbol).history(period="5d", interval=tf)
        if df.empty or len(df) < 50: return []
        
        df = calculate_indicators(df)
        last = df.iloc[-1] 
        prev = df.iloc[-2] 
        
        current_price = last['Close']
        atr = last['ATR']
        decimals = 5 if current_price < 10 else 2
        setups = []

        # ==========================================
        # SNIPER RISK MANAGEMENT (1:3 RISK TO REWARD)
        # ==========================================
        # Stop Loss = 0.4 ATR (Tight risk)
        # Take Profit = 1.2 ATR (Exactly 3x the risk)
        sl_distance = atr * 0.4
        tp_distance = atr * 1.2 

        # ==========================================
        # STRATEGY 1: RSI Sniper Pullback
        # ==========================================
        if prev['EMA_20'] > prev['EMA_50'] and 35 <= prev['RSI'] <= 45:
            entry = round(prev['Close'], decimals)
            if abs(current_price - entry) <= sl_distance:
                setups.append(Setup(
                    timeframe=tf.upper(), direction="BUY", entry=entry,
                    stopLoss=round(entry - sl_distance, decimals),
                    takeProfit=round(entry + tp_distance, decimals),
                    current_price=round(current_price, decimals),
                    strategy_name="RSI Sniper Pullback",
                    strategy_logic="Uptrend pullback. RSI reloaded. Price at exact entry."
                ))
                
        elif prev['EMA_20'] < prev['EMA_50'] and 55 <= prev['RSI'] <= 65:
            entry = round(prev['Close'], decimals)
            if abs(current_price - entry) <= sl_distance:
                setups.append(Setup(
                    timeframe=tf.upper(), direction="SELL", entry=entry,
                    stopLoss=round(entry + sl_distance, decimals),
                    takeProfit=round(entry - tp_distance, decimals),
                    current_price=round(current_price, decimals),
                    strategy_name="RSI Sniper Pullback",
                    strategy_logic="Downtrend pullback. RSI reloaded. Price at exact entry."
                ))

        # ==========================================
        # STRATEGY 2: Momentum Breakout
        # ==========================================
        if current_price > prev['High']:
            entry = round(prev['High'], decimals)
            if abs(current_price - entry) <= sl_distance:
                setups.append(Setup(
                    timeframe=tf.upper(), direction="BUY", entry=entry,
                    stopLoss=round(entry - sl_distance, decimals),
                    takeProfit=round(entry + tp_distance, decimals),
                    current_price=round(current_price, decimals),
                    strategy_name="Momentum Breakout",
                    strategy_logic="Price breaking structure. Entering at the breakout point."
                ))
                
        elif current_price < prev['Low']:
            entry = round(prev['Low'], decimals)
            if abs(current_price - entry) <= sl_distance:
                setups.append(Setup(
                    timeframe=tf.upper(), direction="SELL", entry=entry,
                    stopLoss=round(entry + sl_distance, decimals),
                    takeProfit=round(entry - tp_distance, decimals),
                    current_price=round(current_price, decimals),
                    strategy_name="Momentum Breakdown",
                    strategy_logic="Price breaking structure. Entering at the breakdown point."
                ))

        return setups
    except Exception as e:
        return []

@app.get("/signals")
def get_signals():
    raw_signals = {}
    for pair, symbol in MARKETS.items():
        all_setups = []
        for tf in TIMEFRAMES:
            setups = get_sniper_setups(pair, symbol, tf)
            all_setups.extend(setups)
            time.sleep(0.1) 
        if all_setups:
            raw_signals[pair] = all_setups

    final_signals = []
    for pair, setups in raw_signals.items():
        buy_count = sum(1 for s in setups if s.direction == "BUY")
        sell_count = sum(1 for s in setups if s.direction == "SELL")
        
        if buy_count > sell_count:
            final_signals.append(Signal(pair=pair, setups=[s for s in setups if s.direction == "BUY"]))
        elif sell_count > buy_count:
            final_signals.append(Signal(pair=pair, setups=[s for s in setups if s.direction == "SELL"]))

    return final_signals

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
