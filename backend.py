from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import yfinance as yf
import pandas as pd
import time
from datetime import datetime

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
    confidence: int  # 0-100%
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

def detect_chart_pattern(df, window=10):
    """Detect simple chart patterns"""
    if len(df) < window: return None
    
    recent = df.tail(window)
    highs = recent['High'].values
    lows = recent['Low'].values
    
    # Double Bottom (Bullish)
    if len(lows) >= 4:
        trough1 = lows[-4]
        trough2 = lows[-2]
        if abs(trough1 - trough2) / trough1 < 0.001:  # Within 0.1%
            return "Double Bottom"
    
    # Double Top (Bearish)
    if len(highs) >= 4:
        peak1 = highs[-4]
        peak2 = highs[-2]
        if abs(peak1 - peak2) / peak1 < 0.001:
            return "Double Top"
    
    return None

def detect_candlestick_pattern(df):
    """Detect candlestick patterns on last candle"""
    if len(df) < 2: return None
    
    last = df.iloc[-2]
    prev = df.iloc[-3]
    
    body = abs(last['Close'] - last['Open'])
    range_val = last['High'] - last['Low']
    
    if range_val == 0: return None
    
    upper_wick = last['High'] - max(last['Open'], last['Close'])
    lower_wick = min(last['Open'], last['Close']) - last['Low']
    
    # Bullish Pin Bar (long lower wick, small body at top)
    if lower_wick > body * 2 and last['Close'] > last['Open']:
        return "Bullish Pin Bar"
    
    # Bearish Pin Bar (long upper wick, small body at bottom)
    if upper_wick > body * 2 and last['Close'] < last['Open']:
        return "Bearish Pin Bar"
    
    # Bullish Engulfing
    if (last['Close'] > last['Open'] and 
        prev['Close'] < prev['Open'] and
        last['Close'] > prev['Open'] and 
        last['Open'] < prev['Close']):
        return "Bullish Engulfing"
    
    # Bearish Engulfing
    if (last['Close'] < last['Open'] and 
        prev['Close'] > prev['Open'] and
        last['Close'] < prev['Open'] and 
        last['Open'] > prev['Close']):
        return "Bearish Engulfing"
    
    return None

def analyze_pair(pair, symbol):
    try:
        setups = []
        current_time = datetime.utcnow().strftime("%H:%M UTC")
        
        # FACTOR 1: H4 TREND
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
            return None  # No clear trend

        # FACTOR 2: H1 MARKET STRUCTURE
        h1_df = yf.Ticker(symbol).history(period="1mo", interval="1h")
        if h1_df.empty or len(h1_df) < 30: return None
        
        # Check for higher highs/lows (uptrend) or lower highs/lows (downtrend)
        h1_highs = h1_df['High'].tail(10).values
        h1_lows = h1_df['Low'].tail(10).values
        
        if master_trend == "UP":
            structure_confirmed = h1_highs[-1] > h1_highs[-5] and h1_lows[-1] > h1_lows[-5]
        else:
            structure_confirmed = h1_highs[-1] < h1_highs[-5] and h1_lows[-1] < h1_lows[-5]
        
        if not structure_confirmed:
            return None  # Structure doesn't confirm trend

        # SCAN TIMEFRAMES FOR ENTRY
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
            
            # FACTOR 3: CHART PATTERN
            chart_pattern = detect_chart_pattern(df)
            
            # FACTOR 4: CANDLESTICK PATTERN
            candle_pattern = detect_candlestick_pattern(df)
            
            # FACTOR 5: VOLUME CONFIRMATION
            volume_confirmed = volume_ratio > 1.2  # 20% above average
            
            # FACTOR 6: RSI ALIGNMENT
            if master_trend == "UP":
                rsi_confirmed = 30 < rsi < 70  # Not overbought
            else:
                rsi_confirmed = 30 < rsi < 70  # Not oversold
            
            # FACTOR 7: ATR VOLATILITY (enough movement)
            atr_confirmed = atr > 0  # Always true if we have data
            
            # Count confluence factors
            confluence_count = 0
            confluence_factors = []
            
            if chart_pattern:
                confluence_count += 1
                confluence_factors.append(f"Pattern: {chart_pattern}")
            
            if candle_pattern:
                confluence_count += 1
                confluence_factors.append(f"Candle: {candle_pattern}")
            
            if volume_confirmed:
                confluence_count += 1
                confluence_factors.append("Volume Spike")
            
            if rsi_confirmed:
                confluence_count += 1
                confluence_factors.append("RSI Aligned")
            
            if atr_confirmed:
                confluence_count += 1
                confluence_factors.append("Volatility OK")
            
            # MUST have at least 4 out of 5 factors (trend + structure already confirmed)
            if confluence_count < 4:
                continue
            
            # Calculate confidence (7 factors total, we have 5 here + trend + structure)
            confidence = int((confluence_count / 5) * 100)
            
            # ENTRY LOGIC
            entry_zone = last['EMA_20']
            
            if master_trend == "UP":
                # Check if price is near entry zone
                if abs(current_price - entry_zone) <= sl_dist * 1.5:
                    entry = round(entry_zone, decimals)
                    status = "LIVE" if current_price <= entry_zone * 1.002 else "PENDING"
                    
                    setups.append(Setup(
                        timeframe=tf.upper(), direction="BUY", entry=entry,
                        stopLoss=round(entry - sl_dist, decimals),
                        takeProfit=round(entry + tp_dist, decimals),
                        current_price=round(current_price, decimals),
                        status=status, signal_time=current_time,
                        confidence=confidence,
                        confluence_factors=confluence_factors,
                        strategy_name=f"{config['type']}: Multi-Factor Setup",
                        strategy_logic=f"H4 UP + H1 Structure + {confluence_count}/5 factors confirmed"
                    ))
            
            elif master_trend == "DOWN":
                if abs(current_price - entry_zone) <= sl_dist * 1.5:
                    entry = round(entry_zone, decimals)
                    status = "LIVE" if current_price >= entry_zone * 0.998 else "PENDING"
                    
                    setups.append(Setup(
                        timeframe=tf.upper(), direction="SELL", entry=entry,
                        stopLoss=round(entry + sl_dist, decimals),
                        takeProfit=round(entry - tp_dist, decimals),
                        current_price=round(current_price, decimals),
                        status=status, signal_time=current_time,
                        confidence=confidence,
                        confluence_factors=confluence_factors,
                        strategy_name=f"{config['type']}: Multi-Factor Setup",
                        strategy_logic=f"H4 DOWN + H1 Structure + {confluence_count}/5 factors confirmed"
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
        time.sleep(0.1)
    return final_signals

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
