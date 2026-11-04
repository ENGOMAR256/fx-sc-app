from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MARKETS = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X", 
    "AUDUSD": "AUDUSD=X", "USDCAD": "CAD=X", "NZDUSD": "NZDUSD=X", "USDCHF": "CHF=X",
    "EURGBP": "EURGBP=X", "EURJPY": "EURJPY=X", "GBPJPY": "GBPJPY=X",
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
    risk_level: str
    confluence_factors: List[str]
    strategy_name: str
    strategy_logic: str
    zone_type: str

class Signal(BaseModel):
    pair: str
    setups: List[Setup]

def get_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def detect_classic_patterns(df):
    patterns = []
    if len(df) < 15: return patterns
    lows = df['Low'].tail(15).values
    highs = df['High'].tail(15).values
    if len(lows) >= 5:
        if abs(lows[-5] - lows[-1]) / lows[-5] < 0.0015 and lows[-3] > lows[-5]: patterns.append("Double Bottom")
        if abs(highs[-5] - highs[-1]) / highs[-5] < 0.0015 and highs[-3] < highs[-5]: patterns.append("Double Top")
    if len(highs) >= 7:
        if highs[-6] < highs[-4] and highs[-2] < highs[-4] and abs(highs[-6] - highs[-2]) / highs[-4] < 0.002: patterns.append("Head and Shoulders")
    if len(df) >= 5:
        last_5 = df.tail(5)
        if last_5.iloc[0]['Is_Marubozu'] and last_5.iloc[0]['Is_Bullish'] and all(last_5.iloc[1:]['Body'] < last_5.iloc[1:]['Range'] * 0.4): patterns.append("Bull Flag")
        elif last_5.iloc[0]['Is_Marubozu'] and last_5.iloc[0]['Is_Bearish'] and all(last_5.iloc[1:]['Body'] < last_5.iloc[1:]['Range'] * 0.4): patterns.append("Bear Flag")
    return patterns

def scan_single_pair(pair, symbol):
    try:
        setups = []
        current_time = datetime.utcnow().strftime("%H:%M UTC")
        
        # 1. H4 MARKET STRUCTURE & FAKEOUTS
        h4_df = yf.Ticker(symbol).history(period="3mo", interval="4h")
        if h4_df.empty or len(h4_df) < 50: return None
        
        h4_df['Swing High'] = h4_df['High'] == h4_df['High'].rolling(window=5, center=True).max()
        h4_df['Swing Low'] = h4_df['Low'] == h4_df['Low'].rolling(window=5, center=True).min()
        recent_highs = h4_df[h4_df['Swing High'] == True]['High'].tail(3).tolist()
        recent_lows = h4_df[h4_df['Swing Low'] == True]['Low'].tail(3).tolist()
        
        if len(recent_highs) < 2 or len(recent_lows) < 2: return None
        
        last_h4 = h4_df.iloc[-1]
        h4_fakeout_bull = (last_h4['High'] > max(recent_highs[-2:]) and last_h4['Close'] < max(recent_highs[-2:]))
        h4_fakeout_bear = (last_h4['Low'] < min(recent_lows[-2:]) and last_h4['Close'] > min(recent_lows[-2:]))
        
        if recent_highs[-1] > recent_highs[-2] and recent_lows[-1] > recent_lows[-2]: master_trend = "BULLISH"
        elif recent_highs[-1] < recent_highs[-2] and recent_lows[-1] < recent_lows[-2]: master_trend = "BEARISH"
        else: return None

        # 2. SCAN TIMEFRAMES
        timeframes_config = [
            {"tf": "15m", "period": "5d",  "type": "Scalping",    "sl_mult": 1.0, "tp_mult": 3.0},
            {"tf": "30m", "period": "10d", "type": "Day Trading", "sl_mult": 1.2, "tp_mult": 3.6},
            {"tf": "1h",  "period": "1mo", "type": "Intraday",    "sl_mult": 1.5, "tp_mult": 4.5},
            {"tf": "4h",  "period": "3mo", "type": "Swing",       "sl_mult": 2.0, "tp_mult": 6.0}
        ]

        for config in timeframes_config:
            tf = config["tf"]
            df = yf.Ticker(symbol).history(period=config["period"], interval=tf)
            if df.empty or len(df) < 30: continue
            
            df['ATR'] = get_atr(df)
            df['Body'] = abs(df['Close'] - df['Open'])
            df['Range'] = df['High'] - df['Low']
            df['Is_Marubozu'] = df['Body'] > (df['Range'] * 0.7)
            df['Is_Base'] = df['Body'] < (df['Range'] * 0.3)
            df['Is_Bullish'] = df['Close'] > df['Open']
            df['Is_Bearish'] = df['Close'] < df['Open']
            
            classic_patterns = detect_classic_patterns(df)
            
            # Candle Indexing: c0 is live, c1 is last closed, c2 is prev, etc.
            c5, c4, c3, c2, c1, c0 = df.iloc[-6], df.iloc[-5], df.iloc[-4], df.iloc[-3], df.iloc[-2], df.iloc[-1]
            current_price = c0['Close']
            atr = c1['ATR']
            decimals = 5 if current_price < 10 else 2
            
            sl_dist = atr * config["sl_mult"]
            tp_dist = atr * config["tp_mult"]
            
            score = 0
            factors = []
            factors.append(f"H4: {master_trend}")
            if h4_fakeout_bull: score += 10; factors.append("H4 Bull Fakeout")
            if h4_fakeout_bear: score += 10; factors.append("H4 Bear Fakeout")
            if classic_patterns: score += 15; factors.append(f"Pattern: {', '.join(classic_patterns)}")

            pattern_found, zone_high, zone_low, entry_price, zone_type = None, 0, 0, 0, ""

            # --- SETUP 1: CORE SND PATTERNS (RBR, DBD, RBD, DBR) ---
            # Strict 3-candle Marubozu-Base-Marubozu logic
            if not pattern_found:
                if master_trend == "BULLISH":
                    # DBR (Drop-Base-Rally) -> Reversal/Demand
                    if c3['Is_Bearish'] and c3['Is_Marubozu'] and c2['Is_Base'] and c1['Is_Bullish'] and c1['Is_Marubozu']:
                        pattern_found, zone_high, zone_low = "DBR (Demand Zone)", c2['High'], c2['Low']
                        entry_price, zone_type = zone_high, "Lower Zone (Demand)"
                        score += 30; factors.append("Core DBR Pattern")
                    # RBR (Rally-Base-Rally) -> Continuation/Mid Zone
                    elif c3['Is_Bullish'] and c3['Is_Marubozu'] and c2['Is_Base'] and c1['Is_Bullish'] and c1['Is_Marubozu']:
                        pattern_found, zone_high, zone_low = "RBR (Continuation)", c2['High'], c2['Low']
                        entry_price, zone_type = zone_high, "Mid Zone (Continuation)"
                        score += 25; factors.append("Core RBR Pattern")
                        
                elif master_trend == "BEARISH":
                    # RBD (Rally-Base-Drop) -> Reversal/Supply
                    if c3['Is_Bullish'] and c3['Is_Marubozu'] and c2['Is_Base'] and c1['Is_Bearish'] and c1['Is_Marubozu']:
                        pattern_found, zone_high, zone_low = "RBD (Supply Zone)", c2['High'], c2['Low']
                        entry_price, zone_type = zone_low, "Top Zone (Supply)"
                        score += 30; factors.append("Core RBD Pattern")
                    # DBD (Drop-Base-Drop) -> Continuation/Mid Zone
                    elif c3['Is_Bearish'] and c3['Is_Marubozu'] and c2['Is_Base'] and c1['Is_Bearish'] and c1['Is_Marubozu']:
                        pattern_found, zone_high, zone_low = "DBD (Continuation)", c2['High'], c2['Low']
                        entry_price, zone_type = zone_low, "Mid Zone (Continuation)"
                        score += 25; factors.append("Core DBD Pattern")

            # --- SETUP 2: ENGULFING WITH MOMENTUM ---
            # "Look for engulfing that is followed by next marubozu"
            if not pattern_found:
                if master_trend == "BULLISH" and c2['Is_Bearish'] and c1['Is_Bullish'] and c1['Close'] > c2['Open'] and c1['Open'] < c2['Close']:
                    if c0['Is_Bullish'] and c0['Is_Marubozu']:
                        pattern_found, zone_high, zone_low = "Bullish Engulfing + Momentum", c2['Low'], c1['Low']
                        entry_price, zone_type = c1['Close'], "Lower Zone"
                        score += 20; factors.append("Engulfing + Next Marubozu")
                elif master_trend == "BEARISH" and c2['Is_Bullish'] and c1['Is_Bearish'] and c1['Close'] < c2['Open'] and c1['Open'] > c2['Close']:
                    if c0['Is_Bearish'] and c0['Is_Marubozu']:
                        pattern_found, zone_high, zone_low = "Bearish Engulfing + Momentum", c1['High'], c2['High']
                        entry_price, zone_type = c1['Close'], "Top Zone"
                        score += 20; factors.append("Engulfing + Next Marubozu")

            # --- SETUP 3: QM (QUASIMODO) + MPL + FTR ---
            if not pattern_found:
                if master_trend == "BULLISH":
                    if c4['Is_Bullish'] and c3['Is_Bearish'] and c2['Is_Bearish'] and c2['Low'] < c4['Low']:
                        if c1['Is_Bullish'] and c1['Is_Marubozu'] and c1['Close'] > c4['High']:
                            pattern_found, zone_high, zone_low = "QM Buy (MPL)", c2['High'], c2['Low']
                            entry_price, zone_type = zone_high, "Lower Zone (Demand)"
                            score += 30; factors.append("QM + MPL Breakout")
                elif master_trend == "BEARISH":
                    if c4['Is_Bearish'] and c3['Is_Bullish'] and c2['Is_Bullish'] and c2['High'] > c4['High']:
                        if c1['Is_Bearish'] and c1['Is_Marubozu'] and c1['Close'] < c4['Low']:
                            pattern_found, zone_high, zone_low = "QM Sell (MPL)", c2['High'], c2['Low']
                            entry_price, zone_type = zone_low, "Top Zone (Supply)"
                            score += 30; factors.append("QM + MPL Breakout")

            # --- SETUP 4: FAKEOUTS (Strict Marubozu Rule) ---
            if not pattern_found:
                if master_trend == "BULLISH" and c2['Low'] < c4['Low'] and c1['Is_Bullish'] and c1['Is_Marubozu']:
                    if c1['Close'] > c4['High']:
                        pattern_found, zone_high, zone_low = "QM Support Fakeout", c2['High'], c2['Low']
                        entry_price, zone_type = zone_high, "Lower Zone"
                        score += 25; factors.append("Fakeout + Marubozu Break")
                elif master_trend == "BEARISH" and c2['High'] > c4['High'] and c1['Is_Bearish'] and c1['Is_Marubozu']:
                    if c1['Close'] < c4['Low']:
                        pattern_found, zone_high, zone_low = "PML Engulfing Fakeout", c2['High'], c2['Low']
                        entry_price, zone_type = zone_low, "Top Zone"
                        score += 25; factors.append("Fakeout + Marubozu Break")

            # --- SETUP 5: SNR FLIP ---
            if not pattern_found:
                swing_high = max(c4['High'], c3['High'], c2['High'])
                swing_low = min(c4['Low'], c3['Low'], c2['Low'])
                if master_trend == "BULLISH" and c1['Close'] > swing_high and c0['Low'] <= swing_high * 1.001 and c0['Close'] > c0['Open']:
                    pattern_found, zone_high, zone_low = "SNR Flip", swing_high * 1.002, swing_high * 0.998
                    entry_price, zone_type = zone_high, "Lower Zone"
                    score += 15; factors.append("SNR Flip")
                elif master_trend == "BEARISH" and c1['Close'] < swing_low and c0['High'] >= swing_low * 0.999 and c0['Close'] < c0['Open']:
                    pattern_found, zone_high, zone_low = "SNR Flip", swing_low * 1.002, swing_low * 0.998
                    entry_price, zone_type = zone_low, "Top Zone"
                    score += 15; factors.append("SNR Flip")

            # --- EXECUTE & RISK LEVEL ---
            if pattern_found:
                is_buy = "BULLISH" in master_trend or "Buy" in pattern_found or "RBR" in pattern_found or "DBR" in pattern_found
                
                recent_closes = df['Close'].tail(20).values[:-1]
                if is_buy: touches = sum(1 for p in recent_closes if p <= zone_high * 1.002 and p >= zone_low * 0.998)
                else: touches = sum(1 for p in recent_closes if p >= zone_low * 0.998 and p <= zone_high * 1.002)
                
                risk_level = "Low Risk (High Prob)" if touches == 0 else ("Medium Risk" if touches <= 2 else "High Risk")
                if risk_level == "Low Risk (High Prob)": score += 10

                if is_buy:
                    if current_price <= zone_high * 1.002 and current_price >= zone_low * 0.998:
                        status = "LIVE"; score += 40; factors.append("Reacting at Demand")
                    elif current_price > zone_high:
                        status = "PENDING"; score += 20; factors.append("Approaching Demand")
                    else: continue
                    setups.append(Setup(timeframe=tf.upper(), direction="BUY", entry=round(entry_price, decimals),
                        stopLoss=round(zone_low - (atr * 0.5), decimals), takeProfit=round(entry_price + tp_dist, decimals),
                        current_price=round(current_price, decimals), status=status, signal_time=current_time,
                        confidence=score, risk_level=risk_level, confluence_factors=factors,
                        strategy_name=f"SND: {pattern_found}", strategy_logic=f"Score: {score}/100. {master_trend} H4.", zone_type=zone_type))
                else:
                    if current_price >= zone_low * 0.998 and current_price <= zone_high * 1.002:
                        status = "LIVE"; score += 40; factors.append("Reacting at Supply")
                    elif current_price < zone_low:
                        status = "PENDING"; score += 20; factors.append("Approaching Supply")
                    else: continue
                    setups.append(Setup(timeframe=tf.upper(), direction="SELL", entry=round(entry_price, decimals),
                        stopLoss=round(zone_high + (atr * 0.5), decimals), takeProfit=round(entry_price - tp_dist, decimals),
                        current_price=round(current_price, decimals), status=status, signal_time=current_time,
                        confidence=score, risk_level=risk_level, confluence_factors=factors,
                        strategy_name=f"SND: {pattern_found}", strategy_logic=f"Score: {score}/100. {master_trend} H4.", zone_type=zone_type))

        if setups: return Signal(pair=pair, setups=setups)
        return None
    except Exception: return None

@app.get("/signals")
def get_signals():
    final_signals = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(lambda item: scan_single_pair(item[0], item[1]), MARKETS.items())
        for result in results:
            if result: final_signals.append(result)
    return final_signals

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
