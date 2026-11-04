from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import yfinance as yf
import pandas as pd
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

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
    status: str
    signal_time: str
    confidence: int
    confluence_factors: List[str]
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

def get_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def scan_single_pair(pair_symbol_tuple):
    pair, symbol = pair_symbol_tuple
    try:
        setups = []
        current_time = datetime.utcnow().strftime("%H:%M UTC")
        
        # 1. FAST TREND CHECK (H4)
        h4_df = yf.Ticker(symbol).history(period="3mo", interval="4h")
        if h4_df.empty or len(h4_df) < 50: return None
        
        h4_df['EMA_50'] = h4_df['Close'].ewm(span=50, adjust=False).mean()
        h4_df['EMA_200'] = h4_df['Close'].ewm(span=200, adjust=False).mean()
        last_h4 = h4_df.iloc[-2]
        
        if last_h4['EMA_50'] > last_h4['EMA_200'] * 1.001: master_trend = "UP"
        elif last_h4['EMA_50'] < last_h4['EMA_200'] * 0.999: master_trend = "DOWN"
        else: return None

        # 2. SMART ENTRY SCAN (H1)
        h1_df = yf.Ticker(symbol).history(period="1mo", interval="1h")
        if h1_df.empty or len(h1_df) < 30: return None
        
        h1_df['ATR'] = get_atr(h1_df)
        h1_df['EMA_20'] = h1_df['Close'].ewm(span=20, adjust=False).mean()
        h1_df['RSI'] = get_rsi(h1_df)
        h1_df['Vol_SMA'] = h1_df['Volume'].rolling(20).mean()
        
        last = h1_df.iloc[-2]
        live = h1_df.iloc[-1]
        
        current_price = live['Close']
        atr = last['ATR']
        rsi = last['RSI']
        volume_ratio = last['Volume'] / last['Vol_SMA'] if last['Vol_SMA'] > 0 else 1
        decimals = 5 if current_price < 10 else 2
        
        sl_dist = atr * 1.5
        tp_dist = atr * 4.5
        
        score = 0
        factors = []
        entry_zone = last['EMA_20']
        
        # Factor 1: H4 Trend (20 pts)
        score += 20
        factors.append(f"H4 Trend: {master_trend}")
        
        # Factor 2: RSI Alignment (20 pts)
        rsi_ok = False
        if master_trend == "UP" and 40 < rsi < 80: 
            rsi_ok = True; score += 20; factors.append(f"RSI Bullish ({round(rsi,1)})")
        elif master_trend == "DOWN" and 20 < rsi < 60: 
            rsi_ok = True; score += 20; factors.append(f"RSI Bearish ({round(rsi,1)})")
            
        # Factor 3: Volume (15 pts)
        if volume_ratio > 1.0: 
            score += 15; factors.append("Volume Normal/High")
            
        # Factor 4: Candle Shape (15 pts)
        candle_range = last['High'] - last['Low']
        if candle_range > 0:
            body = abs(last['Close'] - last['Open'])
            if body > candle_range * 0.5: 
                score += 15; factors.append("Strong Candle Body")

        # Factor 5: ENTRY TYPE (30 pts) - THIS IS THE KEY TO MORE PAIRS!
        is_valid_entry = False
        setup_type = ""
        
        # Type A: Pullback (Price near EMA)
        if master_trend == "UP" and last['Low'] <= entry_zone * 1.005 and last['Close'] > entry_zone:
            is_valid_entry = True; score += 30; setup_type = "Pullback Bounce"
        elif master_trend == "DOWN" and last['High'] >= entry_zone * 0.995 and last['Close'] < entry_zone:
            is_valid_entry = True; score += 30; setup_type = "Pullback Rejection"
            
        # Type B: Momentum (Price trending away from EMA with strength)
        elif master_trend == "UP" and last['Close'] > entry_zone and last['Close'] > last['Open'] and rsi > 50:
            is_valid_entry = True; score += 30; setup_type = "Momentum Continuation"
        elif master_trend == "DOWN" and last['Close'] < entry_zone and last['Close'] < last['Open'] and rsi < 50:
            is_valid_entry = True; score += 30; setup_type = "Momentum Breakdown"

        # ONLY SHOW IF SCORE IS 65+ (Smart but inclusive)
        if score >= 65 and is_valid_entry and rsi_ok:
            if master_trend == "UP":
                entry = round(last['Close'], decimals)
                status = "LIVE"
                setups.append(Setup(
                    timeframe="H1", direction="BUY", entry=entry,
                    stopLoss=round(entry - sl_dist, decimals),
                    takeProfit=round(entry + tp_dist, decimals),
                    current_price=round(current_price, decimals),
                    status=status, signal_time=current_time,
                    confidence=score, confluence_factors=factors,
                    strategy_name=f"Smart {setup_type}",
                    strategy_logic=f"Score: {score}/100. {master_trend} trend confirmed."
                ))
            elif master_trend == "DOWN":
                entry = round(last['Close'], decimals)
                status = "LIVE"
                setups.append(Setup(
                    timeframe="H1", direction="SELL", entry=entry,
                    stopLoss=round(entry + sl_dist, decimals),
                    takeProfit=round(entry - tp_dist, decimals),
                    current_price=round(current_price, decimals),
                    status=status, signal_time=current_time,
                    confidence=score, confluence_factors=factors,
                    strategy_name=f"Smart {setup_type}",
                    strategy_logic=f"Score: {score}/100. {master_trend} trend confirmed."
                ))

        if setups:
            return Signal(pair=pair, setups=setups)
        return None

    except Exception:
        return None

@app.get("/signals")
def get_signals():
    final_signals = []
    # MULTI-THREADING: Scans 5 pairs at the exact same time for blazing speed!
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(scan_single_pair, MARKETS.items())
        for result in results:
            if result:
                final_signals.append(result)
    return final_signals

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
