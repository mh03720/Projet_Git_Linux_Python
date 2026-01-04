import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- CONFIGURATION ---
TICKER = "ENGI.PA"
LOG_FILE = "/home/ubuntu/daily_report_log.csv"

def calculate_daily_volatility(df):
    """Calcule la Volatilité JOURNALIÈRE moyenne sur l'historique"""
    returns = df['Close'].pct_change().dropna()
    if len(returns) > 0:
        # Écart-type des rendements quotidiens 
        
        return returns.std()
    else:
        return 0

def job():
    print(f"Récupération des données pour {TICKER}...")
    # On garde 1 an d'historique pour avoir une statistique fiable
    data = yf.download(TICKER, period="1y", interval="1d", progress=False)

    if data.empty:
        print("Erreur: Pas de données récupérées.")
        return

    # Gestion du format MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # 1. Calcul de la Volatilité Journalière
    vol_day = calculate_daily_volatility(data)

    # 2. Récupération des données DU JOUR
    last_row = data.iloc[-1]
    last_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    open_p = float(last_row['Open'])
    close_p = float(last_row['Close'])
    high_p = float(last_row['High'])
    low_p = float(last_row['Low'])

    # 3. Calcul du Max Drawdown DU JOUR 
    if high_p > 0:
        daily_max_dd = (low_p - high_p) / high_p
    else:
        daily_max_dd = 0.0

    
    # Ordre: Date, Ticker, Open, Close, High, Low, Volatility_Daily, Max_Drawdown_Day
    row = f"{last_date},{TICKER},{open_p:.2f},{close_p:.2f},{high_p:.2f},{low_p:.2f},{vol_day:.2%},{daily_max_dd:.2%}\n"
    
    # Écriture dans le fichier
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a") as f:
        if not file_exists:
            # En-tête mis à jour
            f.write("Date,Ticker,Open,Close,High,Low,Volatility_Daily,Max_Drawdown_Day\n")
        f.write(row)

    print(f"Rapport généré : {row.strip()}")

if __name__ == "__main__":
    job()