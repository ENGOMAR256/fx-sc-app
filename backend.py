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
    status: str # PENDING or LIVE
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

def scan_single_pair(pair, symbol):
    try:
        setups = []
        current_time = datetime.utcnow().strftime("%H:%M UTC")
        
        # 1. MASTER TREND (H4)
        h4_df = yf.Ticker(symbol).history(period="3mo", interval="4h")
        if h4_df.empty or len(h4_df) < 50: return None
        
        h4_df['EMA_50'] = h4_df['Close'].ewm(span=50, adjust=False).mean()
        h4_df['EMA_200'] = h4_df['Close'].ewm(span=200, adjust=False).mean()
        last_h4 = h4_df.iloc[-2]
        
        if last_h4['EMA_50'] > last_h4['EMA_200'] * 1.001: master_trend = "UP"
        elif last_h4['EMA_50'] < last_h4['EMA_200'] * 0.999: master_trend = "DOWN"
        else: return None

        # 2. SCAN ALL TIMEFRAMES (Scalping, Day, Intraday, Swing)
        timeframes_config = [
            {"tf": "15m", "period": "5d",  "type": "Scalping",    "sl_mult": 1.0,  "tp_mult": 3.0},
            {"tf": "30m", "period": "10d", "type": "Day Trading", "sl_mult": 1.2,  "tp_mult": 3.6},
            {"tf": "1h",  "period": "1mo", "type": "Intraday",    "sl_mult": 1.5,  "tp_mult": 4.5},
            {"tf": "4h",  "period": "3mo", "type": "Swing",       "sl_mult": 2.0,  "tp_mult": 6.0}
        ]

        for config in timeframes_config:
            tf = config["tf"]
            df = yf.Ticker(symbol).history(period=config["period"], interval=tf)
            if df.empty or len(df) < 30: continue
            
            df['ATR'] = get_atr(df)
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['RSI'] = get_rsi(df)
            df['Vol_SMA'] = df['Volume'].rolling(20).mean()
            
            last = df.iloc[-2]
            live = df.iloc[-1]
            
            current_price = live['Close']
            atr = last['ATR']
            rsi = last['RSI']
            volume_ratio = last['Volume'] / last['Vol_SMA'] if last['Vol_SMA'] > 0 else 1
            decimals = 5 if current_price < 10 else 2
            
            sl_dist = atr * config["sl_mult"]
            tp_dist = atr * config["tp_mult"]
            entry_zone = last['EMA_20']
            
            # SCORING SYSTEM
            score = 0
            factors = []
            
            score += 20; factors.append(f"H4 Trend: {master_trend}")
            
            is_pullback = False
            if master_trend == "UP" and last['Low'] <= entry_zone * 1.003 and last['Close'] > entry_zone:
                is_pullback = True; score += 20; factors.append(f"{tf.upper()} Pullback to 20 EMA")
            elif master_trend == "DOWN" and last['High'] >= entry_zone * 0.997 and last['Close'] < entry_zone:
                is_pullback = True; score += 20; factors.append(f"{tf.upper()} Pullback to 20 EMA")
                
            candle_range = last['High'] - last['Low']
            if candle_range > 0:
                lower_wick = min(last['Open'], last['Close']) - last['Low']
                upper_wick = last['High'] - max(last['Open'], last['Close'])
                if master_trend == "UP" and lower_wick > candle_range * 0.3:
                    score += 20; factors.append("Bullish Rejection Wick")
                elif master_trend == "DOWN" and upper_wick > candle_range * 0.3:
                    score += 20; factors.append("Bearish Rejection Wick")
                    
            if 30 < rsi < 70: score += 20; factors.append(f"RSI Healthy ({round(rsi,1)})")
            if volume_ratio > 1.0: score += 20; factors.append("Volume Confirmed")

            # PENDING vs LIVE LOGIC
            if score >= 60 and is_pullback:
                if master_trend == "UP":
                    entry = round(entry_zone, decimals)
                    # If price is still above the entry zone, it is PENDING. If it touched it, it is LIVE.
                    status = "PENDING" if current_price > entry * 1.002 else "LIVE"
                    
                    setups.append(Setup(
                        timeframe=tf.upper(), direction="BUY", entry=entry,
                        stopLoss=round(entry - sl_dist, decimals),
                        takeProfit=round(entry + tp_dist, decimals),
                        current_price=round(current_price, decimals),
                        status=status, signal_time=current_time,
                        confidence=score, confluence_factors=factors,
                        strategy_name=f"{config['type']} ({tf.upper()})",
                        strategy_logic=f"Score: {score}/100. {master_trend} trend pullback."
                    ))
                
                elif master_trend == "DOWN":
                    entry = round(entry_zone, decimals)
                    status = "PENDING" if current_price < entry * 0.998 else "LIVE"
                    
                    setups.append(Setup(
                        timeframe=tf.upper(), direction="SELL", entry=entry,
                        stopLoss=round(entry + sl_dist, decimals),
                        takeProfit=round(entry - tp_dist, decimals),
                        current_price=round(current_price, decimals),
                        status=status, signal_time=current_time,
                        confidence=score, confluence_factors=factors,
                        strategy_name=f"{config['type']} ({tf.upper()})",
                        strategy_logic=f"Score: {score}/100. {master_trend} trend pullback."
                    ))

        if setups:
            return Signal(pair=pair, setups=setups)
        return None
    except Exception:
        return None

@app.get("/signals")
def get_signals():
    final_signals = []
    # MULTI-THREADING: Scans 6 pairs at the exact same time for blazing speed!
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = executor.map(lambda item: scan_single_pair(item[0], item[1]), MARKETS.items())
        for result in results:
            if result:
                final_signals.append(result)
    return final_signals

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
