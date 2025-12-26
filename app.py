import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh


st.set_page_config(page_title="Quant A - Professional Terminal", layout="wide")
st_autorefresh(interval=5 * 60 * 1000, key="datarefresh") # Refresh 5 min


def fetch_data(ticker, period):
    try:
        data = yf.download(ticker, period=period, interval="1d")
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): 
            data.columns = data.columns.get_level_values(0)
        return data
    except: return None


def strategy_sma(df, short_w, long_w):
    d = df.copy()
    d['SMA_S'] = d['Close'].rolling(window=short_w).mean()
    d['SMA_L'] = d['Close'].rolling(window=long_w).mean()
    d['Signal'] = np.where(d['SMA_S'] > d['SMA_L'], 1, 0)
    return d

def strategy_rsi(df, window=14):
    d = df.copy()
    delta = d['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    d['RSI'] = 100 - (100 / (1 + rs))
    d['Signal'] = np.where(d['RSI'] < 30, 1, np.where(d['RSI'] > 70, 0, np.nan))
    d['Signal'] = d['Signal'].ffill().fillna(0)
    return d

def get_advanced_metrics(df):
    returns = df['Close'].pct_change().dropna()
    strat_returns = df['Signal'].shift(1) * returns
    
  
    cum_strategy = (1 + strat_returns.fillna(0)).cumprod()
    cum_asset = (1 + returns.fillna(0)).cumprod()
    
  
    sharpe = (strat_returns.mean() / strat_returns.std()) * np.sqrt(252) if strat_returns.std() != 0 else 0
    
   
    peak = cum_strategy.cummax()
    drawdown = (cum_strategy - peak) / peak
    max_dd = drawdown.min()
    
  
    vol = strat_returns.std() * np.sqrt(252)
   
    win_rate = len(strat_returns[strat_returns > 0]) / len(strat_returns[strat_returns != 0]) if len(strat_returns[strat_returns != 0]) > 0 else 0
    
    return {
        "cum_strat": cum_strategy,
        "cum_asset": cum_asset,
        "sharpe": round(float(sharpe), 2),
        "max_dd": f"{round(float(max_dd * 100), 2)}%",
        "vol": f"{round(float(vol * 100), 2)}%",
        "win_rate": f"{round(win_rate * 100, 1)}%",
        "total_ret": f"{round((cum_strategy.iloc[-1]-1)*100, 2)}%"
    }

# --- BONUS ML PREDICTION (Linear Regression) ---
def get_ml_forecast(df, days_to_forecast=10):
    d_ml = df[['Close']].reset_index()
    d_ml['Time_Index'] = np.arange(len(d_ml))
    
    X = d_ml[['Time_Index']]
    y = d_ml['Close']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Création des dates futures
    last_date = d_ml['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_forecast + 1)]
    future_indices = np.array([len(d_ml) + i for i in range(days_to_forecast)]).reshape(-1, 1)
    
    preds = model.predict(future_indices)
    
    # Calcul d'un intervalle de confiance simplifié (Ecart-type des résidus)
    residuals = y - model.predict(X)
    std_error = np.std(residuals)
    
    return future_dates, preds, std_error

# --- FRONT-END ---
st.sidebar.title("⚙️ Paramètres Quant")
ticker = st.sidebar.text_input("Symbole Asset", "ENGI.PA")
period = st.sidebar.selectbox("Périodicité", ["1y", "2y", "5y", "max"])
strat_choice = st.sidebar.radio("Stratégie de Backtesting", ["SMA Crossover", "RSI Momentum"])

if strat_choice == "SMA Crossover":
    p1 = st.sidebar.slider("SMA Courte", 5, 50, 20)
    p2 = st.sidebar.slider("SMA Longue", 51, 200, 50)
else:
    p1 = st.sidebar.slider("Période RSI", 7, 30, 14)

data = fetch_data(ticker, period)

if data is not None:
    # Calcul Stratégie
    processed = strategy_sma(data, p1, p2) if strat_choice == "SMA Crossover" else strategy_rsi(data, p1)
    metrics = get_advanced_metrics(processed)
    
    # AFFICHAGE METRIQUES (Header)
    st.subheader(f"Analyse Quantitative : {ticker}")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Sharpe Ratio", metrics["sharpe"])
    col2.metric("Max Drawdown", metrics["max_dd"])
    col3.metric("Volatilité (Ann.)", metrics["vol"])
    col4.metric("Win Rate", metrics["win_rate"])
    col5.metric("Total Return", metrics["total_ret"])

    # GRAPHIQUE PRINCIPAL
    fig = go.Figure()
    # Courbe Asset
    fig.add_trace(go.Scatter(x=data.index, y=metrics["cum_asset"], name="Asset Buy & Hold", line=dict(color='rgba(200, 200, 200, 0.5)', dash='dash')))
    # Courbe Stratégie
    fig.add_trace(go.Scatter(x=data.index, y=metrics["cum_strat"], name=f"Stratégie {strat_choice}", line=dict(color='#00CCFF', width=3)))
    
    # BONUS ML FORECAST
    if st.sidebar.checkbox("Activer Prédiction ML (Bonus)"):
        f_dates, f_preds, err = get_ml_forecast(data)
        # Normalisation pour le graph cumulé
        last_val = metrics["cum_strat"].iloc[-1]
        norm_preds = (f_preds / data['Close'].iloc[-1]) * last_val
        
        # Intervalle de confiance
        upper_bound = norm_preds + (err / data['Close'].iloc[-1])
        lower_bound = norm_preds - (err / data['Close'].iloc[-1])
        
        fig.add_trace(go.Scatter(x=f_dates, y=norm_preds, name="Forecast ML (10j)", line=dict(color='orange', width=2)))
        fig.add_trace(go.Scatter(x=f_dates+f_dates[::-1], y=list(upper_bound)+list(lower_bound)[::-1], fill='toself', fillcolor='rgba(255,165,0,0.2)', line=dict(color='rgba(255,255,255,0)'), name="Confiance 68%"))

    fig.update_layout(template="plotly_dark", height=600, hovermode="x unified", title="Comparaison de Performance et Prévisions")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Données indisponibles pour ce Ticker.")