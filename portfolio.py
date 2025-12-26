import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh


AVAILABLE_ASSETS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "MATIC-USD",
    "AC.PA", "ACA.PA", "AI.PA", "AIR.PA", "ALO.PA", "ATO.PA", "BN.PA", "BNP.PA", "CA.PA", "CAP.PA", 
    "CS.PA", "DG.PA", "DSY.PA", "EL.PA", "EN.PA", "ENGI.PA", "ERF.PA", "GLE.PA", "HO.PA", "KER.PA", 
    "LR.PA", "MC.PA", "ML.PA", "ORA.PA", "OR.PA", "PUB.PA", "RI.PA", "RNO.PA", "SAF.PA", "SAN.PA", 
    "SGO.PA", "STLAP.PA", "SU.PA", "SW.PA", "TEP.PA", "TTE.PA", "URW.PA", "VIE.PA", "VIV.PA",
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX", "AMD", "INTC", 
    "JPM", "BAC", "WMT", "PG", "JNJ", "XOM", "CVX", "KO", "PEP", "MCD", "DIS", "NKE",
    "^FCHI", "^GSPC", "^IXIC", "GC=F", "CL=F", "EURUSD=X"
]
AVAILABLE_ASSETS.sort()


def get_current_price(ticker):
    try:
        df = yf.download(ticker, period="1d", interval="1m", progress=False)
        if df.empty: df = yf.download(ticker, period="5d", interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        price = df['Close'].iloc[-1]
        if isinstance(price, pd.Series): price = price.iloc[0]
        return float(price)
    except: return None


def optimize_sma_params(df):
    best_ret = -np.inf
    best_params = (20, 50)
    for s in range(10, 50, 10):
        for l in range(50, 160, 20):
            if s >= l: continue
            d = df.copy()
            d['S'] = d['Close'].rolling(window=s).mean()
            d['L'] = d['Close'].rolling(window=l).mean()
            d['Sig'] = np.where(d['S'] > d['L'], 1, 0)
            ret = d['Close'].pct_change().shift(-1) * d['Sig']
            total_ret = (1 + ret.fillna(0)).cumprod().iloc[-1]
            if total_ret > best_ret:
                best_ret = total_ret
                best_params = (s, l)
    return best_params, best_ret