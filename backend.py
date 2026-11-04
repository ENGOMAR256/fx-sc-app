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

TIMEFRAMES = ["15m", "30m", "1h", "4h"] # Removed 1m/5m to focus on authentic swing structure

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

def find_swings(df, window=5):
    """Find Swing Highs and Swing Lows (Fractals)"""
    df['Swing High'] = df['High'] == df['High'].rolling(window=window, center=True).max()
    df['Swing Low'] = df['Low'] == df['Low'].rolling(window=window, center=True).min()
    return df

def get_smc_setups(pair, symbol, tf):
    try:
        period = "1mo" if tf in ["15m", "30m"] else "6mo"
        df = yf.Ticker(symbol).history(period=period, interval=tf)
        if df.empty or len(df) < 50: return []
        
        df = find_swings(df, window=5)
        
        # Calculate ATR for Stop Loss / Take Profit
        high_low = df['High'] - df['Low']
        tr = high_low.rolling(14).mean()
        df['ATR'] = tr
        last = df.iloc[-1]
        prev = df.iloc[-2]
        current_price = last['Close']
        atr = last['ATR']
        decimals = 5 if current_price < 10 else 2
        setups = []

        # Get recent swing points
        recent_lows = df[df['Swing Low'] == True]['Low'].tail(3).tolist()
        recent_highs = df[df['Swing High'] == True]['High'].tail(3).tolist()

        if len(recent_lows) < 2 or len(recent_highs) < 2: return []

        last_swing_low = recent_lows[-1]
        prev_swing_low = recent_lows[-2]
        last_swing_high = recent_highs[-1]
        prev_swing_high = recent_highs[-2]

        # ==========================================
        # BULLISH SMC SETUP (Liquidity Sweep + BOS)
        # ==========================================
        # 1. Market Structure is Bullish (Higher Highs)
        # 2. Price sweeps below the previous Swing Low (Liquidity Grab)
        # 3. Price closes back above the previous Swing Low (Rejection)
        
        is_bullish_structure = last_swing_high > prev_swing_high
        swept_liquidity = last['Low'] < prev_swing_low
        rejected_sweep = last['Close'] > prev_swing_low and last['Open'] < prev_swing_low
        
        if is_bullish_structure and swept_liquidity and rejected_sweep:
            # Entry: Current price (or limit at the Order Block, simplified here to market on confirmation)
            # Stop Loss: Just below the sweep wick
            # Take Profit: Next major Swing High
            entry = round(current_price, decimals)
            sl = round(last['Low'] - (atr * 0.5), decimals)
            tp = round(last_swing_high + (atr * 1.0), decimals)
            
            setups.append(Setup(
                timeframe=tf.upper(), direction="BUY", 
                entry=entry, stopLoss=sl, takeProfit=tp, 
                current_price=round(current_price, decimals),
                strategy_name="SMC: Liquidity Sweep + BOS", 
                strategy_logic=f"Bullish structure. Price swept {prev_swing_low} (retail stops) and rejected. Banks are buying."
            ))

        # ==========================================
        # BEARISH SMC SETUP (Liquidity Sweep + BOS)
        # ==========================================
        is_bearish_structure = last_swing_low < prev_swing_low
        swept_liquidity = last['High'] > prev_swing_high
        rejected_sweep = last['Close'] < prev_swing_high and last['Open'] > prev_swing_high

        if is_bearish_structure and swept_liquidity and rejected_sweep:
            entry = round(current_price, decimals)
            sl = round(last['High'] + (atr * 0.5), decimals)
            tp = round(last_swing_low - (atr * 1.0), decimals)
            
            setups.append(Setup(
                timeframe=tf.upper(), direction="SELL", 
                entry=entry, stopLoss=sl, takeProfit=tp, 
                current_price=round(current_price, decimals),
                strategy_name="SMC: Liquidity Sweep + BOS", 
                strategy_logic=f"Bearish structure. Price swept {prev_swing_high} (retail stops) and rejected. Banks are selling."
            ))

        return setups
    except Exception as e:
        print(f"Error {pair} {tf}: {e}")
        return []

@app.get("/signals")
def get_signals():
    raw_signals = {}
    for pair, symbol in MARKETS.items():
        all_setups = []
        for tf in TIMEFRAMES:
            setups = get_smc_setups(pair, symbol, tf)
            all_setups.extend(setups)
            time.sleep(0.2) 
        if all_setups:
            raw_signals[pair] = all_setups

    # Conflict Resolution (Keep only the dominant direction)
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
