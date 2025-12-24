import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- CONFIGURATION ---
TICKER = "ENGI.PA"
LOG_FILE = "/home/ubuntu/daily_report_log.csv"

def calculate_metrics(df):
    """Calcule Volatilité et Max Drawdown sur l'historique récupéré"""
    # Rendements journaliers
    returns = df['Close'].pct_change().dropna()

    # 1. Volatilité Annualisée (Standard Deviation * racine(252 jours))
    if len(returns) > 0:
        vol_ann = returns.std() * np.sqrt(252)
    else:
        vol_ann = 0

    # 2. Max Drawdown
    cum_ret = (1 + returns).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_dd = drawdown.min()

    return vol_ann, max_dd

def job():
    # On récupère 1 an de données pour que la Volatilité et le Drawdown aient du sens
    print(f"Récupération des données pour {TICKER}...")
    data = yf.download(TICKER, period="1y", interval="1d", progress=False)

    if data.empty:
        print("Erreur: Pas de données récupérées.")
        return

    # Gestion du format MultiIndex de yfinance (v0.2+)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Calculs
    vol_ann, max_dd = calculate_metrics(data)

    # Valeurs du jour (Dernière ligne)
    last_row = data.iloc[-1]
    last_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    open_p = last_row['Open']
    close_p = last_row['Close']
    high_p = last_row['High']
    low_p = last_row['Low']

    # Formatage de la ligne CSV
    # Ordre: Date, Ticker, Open, Close, High, Low, Volatility, Max_Drawdown
    row = f"{last_date},{TICKER},{open_p:.2f},{close_p:.2f},{high_p:.2f},{low_p:.2f},{vol_ann:.2%},{max_dd:.2%}\n"
    
    # Écriture dans le fichier (Mode 'a' pour append/ajouter à la fin)
    # Création de l'entête si le fichier n'existe pas
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a") as f:
        if not file_exists:
            f.write("Date,Ticker,Open,Close,High,Low,Volatility_Ann,Max_Drawdown\n")
        f.write(row)

    print(f"Rapport généré avec succès dans {LOG_FILE}")
    print(f"Données: {row.strip()}")

if __name__ == "__main__":
    job()