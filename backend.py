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

# ALL TIMEFRAMES USED
TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h", "1d"]

class Setup(BaseModel):
    timeframe: str
    direction: str
    entry: float
    stopLoss: float
    takeProfit: float
    rsi: float
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
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    df['Vol_SMA'] = df['Volume'].rolling(20).mean()
    df['High_20'] = df['High'].rolling(20).max()
    df['Low_20'] = df['Low'].rolling(20).min()
    df['Body_Size'] = (df['Close'] - df['Open']).abs()
    df['Avg_Body'] = df['Body_Size'].rolling(20).mean()
    return df

def get_setups(pair, symbol, tf):
    try:
        df = yf.Ticker(symbol).history(period="3mo", interval=tf)
        if df.empty or len(df) < 200: return []
        df = calculate_indicators(df)
        last = df.iloc[-2] 
        prev = df.iloc[-3]
        current_rsi = last['RSI']
        current_price = last['Close']
        atr = last['ATR']
        decimals = 5 if current_price < 10 else 2
        setups = []

        # ANTI-SCALPING LOGIC: Lower timeframes get wider stops (2.0 ATR) and bigger targets (4.0 ATR)
        # This ensures M5/M15 entries are held for swing moves, not quick scalps.
        is_low_tf = tf in ["5m", "15m"]
        sl_mult = 2.0 if is_low_tf else 1.5
        tp_mult = 4.0 if is_low_tf else 3.0

        # 1. SMC / Liquidity
        if current_rsi < 25:
            setups.append(Setup(timeframe=tf.upper(), direction="BUY", entry=round(current_price, decimals), 
                stopLoss=round(current_price - (atr * sl_mult), decimals), takeProfit=round(current_price + (atr * tp_mult), decimals), 
                rsi=round(current_rsi, 1), current_price=round(current_price, decimals),
                strategy_name="SMC / Liquidity Grab", strategy_logic="Extreme oversold. High probability swing bounce."))
        elif current_rsi > 75:
            setups.append(Setup(timeframe=tf.upper(), direction="SELL", entry=round(current_price, decimals), 
                stopLoss=round(current_price + (atr * sl_mult), decimals), takeProfit=round(current_price - (atr * tp_mult), decimals), 
                rsi=round(current_rsi, 1), current_price=round(current_price, decimals),
                strategy_name="SMC / Liquidity Grab", strategy_logic="Extreme overbought. High probability swing pullback."))

        # 2. Trend Following
        if last['EMA_50'] > last['EMA_200'] and current_price > last['EMA_50'] and last['Low'] <= last['EMA_50'] * 1.002:
            setups.append(Setup(timeframe=tf.upper(), direction="BUY", entry=round(current_price, decimals), 
                stopLoss=round(last['EMA_200'], decimals), takeProfit=round(current_price + (atr * tp_mult), decimals), 
                rsi=round(current_rsi, 1), current_price=round(current_price, decimals),
                strategy_name="Trend Following", strategy_logic="Strong uptrend. Pullback to 50 EMA confirmed."))
        elif last['EMA_50'] < last['EMA_200'] and current_price < last['EMA_50'] and last['High'] >= last['EMA_50'] * 0.998:
            setups.append(Setup(timeframe=tf.upper(), direction="SELL", entry=round(current_price, decimals), 
                stopLoss=round(last['EMA_200'], decimals), takeProfit=round(current_price - (atr * tp_mult), decimals), 
                rsi=round(current_rsi, 1), current_price=round(current_price, decimals),
                strategy_name="Trend Following", strategy_logic="Strong downtrend. Pullback to 50 EMA confirmed."))

        # 3. Momentum Breakout
        if current_price > prev['High_20'] and last['Volume'] > last['Vol_SMA'] * 1.5:
            setups.append(Setup(timeframe=tf.upper(), direction="BUY", entry=round(current_price, decimals), 
                stopLoss=round(current_price - (atr * sl_mult), decimals), takeProfit=round(current_price + (atr * tp_mult), decimals), 
                rsi=round(current_rsi, 1), current_price=round(current_price, decimals),
                strategy_name="Momentum Breakout", strategy_logic="High volume breakout above 20-period high."))
        elif current_price < prev['Low_20'] and last['Volume'] > last['Vol_SMA'] * 1.5:
            setups.append(Setup(timeframe=tf.upper(), direction="SELL", entry=round(current_price, decimals), 
                stopLoss=round(current_price + (atr * sl_mult), decimals), takeProfit=round(current_price - (atr * tp_mult), decimals), 
                rsi=round(current_rsi, 1), current_price=round(current_price, decimals),
                strategy_name="Momentum Breakout", strategy_logic="High volume breakdown below 20-period low."))

        # 4. Supply & Demand
        if prev['Body_Size'] > prev['Avg_Body'] * 1.5 and prev['Close'] > prev['Open']:
            if current_price >= prev['Open'] and current_price <= prev['Close'] * 1.002:
                setups.append(Setup(timeframe=tf.upper(), direction="BUY", entry=round(current_price, decimals), 
                    stopLoss=round(prev['Open'] - (atr * 1.0), decimals), takeProfit=round(current_price + (atr * tp_mult), decimals), 
                    rsi=round(current_rsi, 1), current_price=round(current_price, decimals),
                    strategy_name="Supply & Demand", strategy_logic="Retest of strong bullish demand zone."))
        elif prev['Body_Size'] > prev['Avg_Body'] * 1.5 and prev['Close'] < prev['Open']:
            if current_price <= prev['Open'] and current_price >= prev['Close'] * 0.998:
                setups.append(Setup(timeframe=tf.upper(), direction="SELL", entry=round(current_price, decimals), 
                    stopLoss=round(prev['Open'] + (atr * 1.0), decimals), takeProfit=round(current_price - (atr * tp_mult), decimals), 
                    rsi=round(current_rsi, 1), current_price=round(current_price, decimals),
                    strategy_name="Supply & Demand", strategy_logic="Retest of strong bearish supply zone."))

        return setups
    except: return []

@app.get("/signals")
def get_signals():
    final_signals = []
    for pair, symbol in MARKETS.items():
        all_setups = []
        for tf in TIMEFRAMES:
            setups = get_setups(pair, symbol, tf)
            all_setups.extend(setups)
            time.sleep(0.1)
        if all_setups:
            final_signals.append(Signal(pair=pair, setups=all_setups))
    return final_signals

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)