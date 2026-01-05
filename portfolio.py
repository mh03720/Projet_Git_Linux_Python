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

def module_market_analysis():
        from app import (
        fetch_data as _fetch_data,
        strategy_sma as _strategy_sma,
        strategy_rsi as _strategy_rsi,
        get_advanced_metrics as _get_advanced_metrics,
        get_ml_forecast as _get_ml_forecast,
    )
st.header("Market Analysis (Single Asset)")
ticker = st.sidebar.selectbox("Rechercher un Actif", AVAILABLE_ASSETS, index=AVAILABLE_ASSETS.index("ENGI.PA") if "ENGI.PA" in AVAILABLE_ASSETS else 0)
    
    # Ajout de 1d, 5d, 1mo pour voir le court terme
period = st.sidebar.selectbox("Périodicité", ["1d", "5d", "1mo", "6mo", "1y", "2y", "5y", "max"])
    
strat_choice = st.sidebar.radio("Stratégie", ["SMA Crossover", "RSI Momentum"])

data = _fetch_data(ticker, period)
if data is not None:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Auto-Optimisation")
        default_s, default_l = 20, 50
        if strat_choice == "SMA Crossover":
            if st.sidebar.button("Trouver Meilleurs Paramètres"):
                with st.spinner("Analyse..."):
                    best_p, best_r = optimize_sma_params(data)
                    st.sidebar.success(f"Top: {best_p[0]}/{best_p[1]} (Gain: {(best_r-1)*100:.1f}%)")
                    default_s, default_l = best_p
            p1 = st.sidebar.slider("SMA Courte", 5, 50, default_s)
            p2 = st.sidebar.slider("SMA Longue", 51, 200, default_l)
        else:
            p1 = st.sidebar.slider("RSI Période", 7, 30, 14)
            p2 = None

        cur_price = get_current_price(ticker)
        if cur_price:
            prev = float(data['Close'].iloc[-1])
            var = ((cur_price - prev)/prev)*100
            st.metric(f"Prix Actuel ({ticker})", f"{cur_price:.2f} €/$", f"{var:+.2f} %")

        processed = _strategy_sma(data, p1, p2) if strat_choice == "SMA Crossover" else strategy_rsi(data, p1)
        metrics = _get_advanced_metrics(processed)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Prix Réel", line=dict(color='gray', width=1)))
        init_p = float(data['Close'].iloc[0])
        fig.add_trace(go.Scatter(x=data.index, y=metrics["cum_strat"] * init_p, name="Stratégie", line=dict(color='#00CCFF', width=2)))
        
        if st.sidebar.checkbox("Prédiction ML"):
            dates, preds, err = _get_ml_forecast(data)
            fig.add_trace(go.Scatter(x=dates, y=preds, name="Forecast", line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=list(dates)+list(dates)[::-1], y=list(preds+err)+list(preds-err)[::-1], fill='toself', fillcolor='rgba(255,165,0,0.2)', line=dict(color='rgba(0,0,0,0)')))

        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sharpe", metrics["sharpe"])
        c2.metric("Max DD", metrics["max_dd"])
        c3.metric("Win Rate", metrics["win_rate"])
        c4.metric("Total Ret", metrics["total_ret"])
else: st.error("Données indisponibles (Marché fermé ou Ticker invalide).")

def module_portfolio_sim():
    st.header("Portfolio Simulator")
    default_sel = [x for x in ["MC.PA", "TTE.PA", "BTC-USD"] if x in AVAILABLE_ASSETS]
    tickers = st.sidebar.multiselect("Actifs (Min 3)", AVAILABLE_ASSETS, default=default_sel)
    
    # Choix de l'horizon pour le 5 minutes
    st.sidebar.markdown("---")
    time_mode = st.sidebar.radio("Horizon Temporel", ["Historique (2 ans)", "Live (5 jours / 5 min)"])
    
    if len(tickers) < 3:
        st.warning("Sélectionnez au moins 3 actifs.")
        return

    # Logique pour adapter la période
    if time_mode == "Historique (2 ans)":
        p, i = "2y", "1d"
    else:
        p, i = "5d", "5m" # Mode Live : 5 derniers jours, intervalle 5 minutes

    # Téléchargement des données avec l'intervalle choisi
    data = yf.download(tickers, period=p, interval=i, progress=False)['Close']
    
    if not data.empty:
        # Forward fill pour aligner les cryptos (24/7) et les actions (9-17h) en mode 5m
        data = data.ffill().dropna()
        returns = data.pct_change().dropna()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("Configuration")
            w_type = st.radio("Mode", ["Équipondéré", "Personnalisé"])
            rebalance = st.selectbox("Rééquilibrage", ["Quotidien (Constant Mix)", "Aucun (Buy & Hold)"])
        
        weights = []
        if w_type == "Équipondéré":
            weights = np.array([1/len(tickers)] * len(tickers))
            with col2:
                fig_pie = px.pie(names=tickers, values=weights, title="Allocation Cible")
                fig_pie.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=0))
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            with col2:
                st.markdown("Poids (Score 1-10)")
                cols_sliders = st.columns(3)
                raw_scores = []
                for i, t in enumerate(tickers):
                    score = cols_sliders[i % 3].slider(f"{t}", 1, 10, 5)
                    raw_scores.append(score)
                weights = np.array(raw_scores) / sum(raw_scores)
                fig_pie = px.pie(values=weights, names=tickers, title="Allocation Réelle (%)")
                fig_pie.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=0))
                st.plotly_chart(fig_pie, use_container_width=True)

        if rebalance == "Quotidien (Constant Mix)":
            port_ret = returns.dot(weights)
            cum_port = (1 + port_ret).cumprod() - 1
        else:
            norm_prices = data / data.iloc[0]
            w_prices = norm_prices * weights
            port_val = w_prices.sum(axis=1)
            cum_port = port_val - 1
            port_ret = port_val.pct_change().dropna()
        
        cum_assets = (1 + returns).cumprod() - 1
        tot = cum_port.iloc[-1] * 100
        
       
        vol = port_ret.std() * np.sqrt(252) * 100
        sharpe = (port_ret.mean() / port_ret.std()) * np.sqrt(252)

        k1, k2, k3 = st.columns(3)
        k1.metric("Rendement", f"{tot:+.2f} %")
        k2.metric("Volatilité (Annu.)", f"{vol:.2f} %")
        k3.metric("Sharpe", f"{sharpe:.2f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum_port.index, y=cum_port*100, name="PORTEFEUILLE", line=dict(color='#FF2B2B', width=4)))
        for t in tickers:
            fig.add_trace(go.Scatter(x=cum_assets.index, y=cum_assets[t]*100, name=t, opacity=0.4))
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(px.imshow(returns.corr(), text_auto=True, color_continuous_scale='RdBu_r'), use_container_width=True)
