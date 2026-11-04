from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import yfinance as yf
import pandas as pd
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
    status: str # "PENDING" or "LIVE"
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
        
        # 1. MASTER TREND (H4) - Fast fetch
        h4_df = yf.Ticker(symbol).history(period="3mo", interval="4h")
        if h4_df.empty or len(h4_df) < 50: return None
        
        h4_df['EMA_50'] = h4_df['Close'].ewm(span=50, adjust=False).mean()
        h4_df['EMA_200'] = h4_df['Close'].ewm(span=200, adjust=False).mean()
        last_h4 = h4_df.iloc[-2]
        
        if last_h4['EMA_50'] > last_h4['EMA_200']: master_trend = "UP"
        elif last_h4['EMA_50'] < last_h4['EMA_200']: master_trend = "DOWN"
        else: return None

        # 2. SCAN TIMEFRAMES
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
            
            last = df.iloc[-2]      # Last CLOSED candle (for pattern confirmation)
            live = df.iloc[-1]      # CURRENT FORMING candle (for live price)
            
            current_price = live['Close']
            atr = last['ATR']
            decimals = 5 if current_price < 10 else 2
            
            sl_dist = atr * config["sl_mult"]
            tp_dist = atr * config["tp_mult"]
            
            # The "Entry Zone" is the 20 EMA of the last closed candle
            entry_zone = last['EMA_20']
            
            # BUY LOGIC (Only if H4 is UP)
            if master_trend == "UP":
                # Did the closed candle break structure and pull back?
                prev_high = df['High'].iloc[-6:-2].max()
                broke_structure = last['High'] > prev_high
                touched_zone = last['Low'] <= entry_zone * 1.002
                
                if broke_structure and touched_zone:
                    # PENDING vs LIVE Logic
                    # If live price is still above the zone, it's PENDING (approaching)
                    # If live price is at or below the zone, it's LIVE (reacting)
                    if current_price > entry_zone * 1.002:
                        status = "PENDING"
                        entry = round(entry_zone, decimals)
                    else:
                        status = "LIVE"
                        entry = round(last['Close'], decimals)
                        
                    setups.append(Setup(
                        timeframe=tf.upper(), direction="BUY", entry=entry,
                        stopLoss=round(entry - sl_dist, decimals),
                        takeProfit=round(entry + tp_dist, decimals),
                        current_price=round(current_price, decimals),
                        status=status,
                        strategy_name=f"{config['type']}: BOS Pullback",
                        strategy_logic=f"H4 UP. {tf.upper()} broke structure. Waiting for bounce at 20 EMA."
                    ))

            # SELL LOGIC (Only if H4 is DOWN)
            elif master_trend == "DOWN":
                prev_low = df['Low'].iloc[-6:-2].min()
                broke_structure = last['Low'] < prev_low
                touched_zone = last['High'] >= entry_zone * 0.998
                
                if broke_structure and touched_zone:
                    if current_price < entry_zone * 0.998:
                        status = "PENDING"
                        entry = round(entry_zone, decimals)
                    else:
                        status = "LIVE"
                        entry = round(last['Close'], decimals)
                        
                    setups.append(Setup(
                        timeframe=tf.upper(), direction="SELL", entry=entry,
                        stopLoss=round(entry + sl_dist, decimals),
                        takeProfit=round(entry - tp_dist, decimals),
                        current_price=round(current_price, decimals),
                        status=status,
                        strategy_name=f"{config['type']}: BOS Pullback",
                        strategy_logic=f"H4 DOWN. {tf.upper()} broke structure. Waiting for rejection at 20 EMA."
                    ))

        if setups:
            return Signal(pair=pair, setups=setups)
        return None

    except Exception:
        return None

@app.get("/signals")
def get_signals():
    final_signals = []
    for pair, symbol in MARKETS.items():
        signal = analyze_pair(pair, symbol)
        if signal:
            final_signals.append(signal)
        time.sleep(0.1) # Optimized speed
    return final_signals

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
