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

def get_ema(df, span):
    return df['Close'].ewm(span=span, adjust=False).mean()

def get_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def get_top_down_signal(pair, symbol):
    try:
        # 1. GET 4-HOUR DATA (The Trend Filter)
        h4_df = yf.Ticker(symbol).history(period="6mo", interval="4h")
        if h4_df.empty or len(h4_df) < 200: return None
        
        h4_df['EMA_50'] = get_ema(h4_df, 50)
        h4_df['EMA_200'] = get_ema(h4_df, 200)
        
        # Use the last CLOSED 4H candle to determine trend
        last_h4 = h4_df.iloc[-2] 
        h4_trend = "UP" if last_h4['EMA_50'] > last_h4['EMA_200'] else "DOWN"

        # 2. GET 1-HOUR DATA (The Entry Trigger)
        h1_df = yf.Ticker(symbol).history(period="1mo", interval="1h")
        if h1_df.empty or len(h1_df) < 50: return None
        
        h1_df['EMA_20'] = get_ema(h1_df, 20)
        h1_df['ATR'] = get_atr(h1_df)
        
        # Use the last CLOSED 1H candle for the trigger
        last_h1 = h1_df.iloc[-2]
        current_price = h1_df.iloc[-1]['Close'] # Live price
        atr = last_h1['ATR']
        decimals = 5 if current_price < 10 else 2
        
        setups = []
        
        # Risk Management: 1:3 RR
        sl_distance = atr * 0.5
        tp_distance = atr * 1.5 # 3x the risk

        # ==========================================
        # BUY LOGIC (Only if 4H Trend is UP)
        # ==========================================
        if h4_trend == "UP":
            # Check if the closed 1H candle pulled back to the 20 EMA and rejected it
            # (Low touched or went below EMA20, but Closed ABOVE EMA20)
            ema20_val = last_h1['EMA_20']
            pulled_back = last_h1['Low'] <= ema20_val * 1.002
            rejected = last_h1['Close'] > ema20_val
            
            if pulled_back and rejected:
                entry = round(last_h1['Close'], decimals)
                # Live Proximity: Only show if live price is near the entry
                if abs(current_price - entry) <= sl_distance:
                    setups.append(Setup(
                        timeframe="H1 (4H Trend: UP)", direction="BUY", entry=entry,
                        stopLoss=round(entry - sl_distance, decimals),
                        takeProfit=round(entry + tp_distance, decimals),
                        current_price=round(current_price, decimals),
                        strategy_name="Top-Down Pullback",
                        strategy_logic=f"4H is Bullish. 1H pulled back to 20 EMA ({round(ema20_val, decimals)}) and rejected. Buyers defending trend."
                    ))

        # ==========================================
        # SELL LOGIC (Only if 4H Trend is DOWN)
        # ==========================================
        elif h4_trend == "DOWN":
            ema20_val = last_h1['EMA_20']
            pulled_back = last_h1['High'] >= ema20_val * 0.998
            rejected = last_h1['Close'] < ema20_val
            
            if pulled_back and rejected:
                entry = round(last_h1['Close'], decimals)
                if abs(current_price - entry) <= sl_distance:
                    setups.append(Setup(
                        timeframe="H1 (4H Trend: DOWN)", direction="SELL", entry=entry,
                        stopLoss=round(entry + sl_distance, decimals),
                        takeProfit=round(entry - tp_distance, decimals),
                        current_price=round(current_price, decimals),
                        strategy_name="Top-Down Pullback",
                        strategy_logic=f"4H is Bearish. 1H pulled back to 20 EMA ({round(ema20_val, decimals)}) and rejected. Sellers defending trend."
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
        signal = get_top_down_signal(pair, symbol)
        if signal:
            final_signals.append(signal)
        time.sleep(0.5) # Prevent rate limits
    return final_signals

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
