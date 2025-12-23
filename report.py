import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- CONFIGURATION ---
TICKER = "ENGI.PA"
SAVE_DIR = os.path.join(os.getcwd(), "reports")
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def calculate_metrics(df):
    """Calcule les métriques avancées pour le rapport."""
    # Rendements
    returns = df['Close'].pct_change().dropna()
    
    # Volatilité annualisée (base 252 jours)
    vol_ann = returns.std() * np.sqrt(252)
    
    # Max Drawdown
    cum_ret = (1 + returns).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_dd = drawdown.min()
    
    return vol_ann, max_dd

def generate_full_report():
    print(f"Génération du rapport pour {TICKER}...")
    # On prend 1 an de données pour avoir un Max Drawdown et une Volatilité cohérente
    data = yf.download(TICKER, period="1y")
    
    if not data.empty:
        # Nettoyage si MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        vol_ann, max_dd = calculate_metrics(data)
        
        # Préparation de la ligne de données
        report_data = {
            "Rapport_Date": [datetime.now().strftime("%Y-%m-%d %H:%M")],
            "Asset": [TICKER],
            "Open_Price": [round(float(data['Open'].iloc[-1]), 2)],
            "Close_Price": [round(float(data['Close'].iloc[-1]), 2)],
            "Daily_High": [round(float(data['High'].iloc[-1]), 2)],
            "Daily_Low": [round(float(data['Low'].iloc[-1]), 2)],
            "Annual_Vol": [f"{round(vol_ann * 100, 2)}%"],
            "Max_Drawdown": [f"{round(max_dd * 100, 2)}%"],
            "Daily_Return": [f"{round(float(data['Close'].pct_change().iloc[-1] * 100), 2)}%"]
        }
        
        df_report = pd.DataFrame(report_data)
        
        # Sauvegarde avec nom unique par jour
        file_name = f"daily_report_{datetime.now().strftime('%Y%m%d')}.csv"
        path = os.path.join(SAVE_DIR, file_name)
        
        df_report.to_csv(path, index=False)
        print(f"Succès : Rapport sauvegardé sous {path}")

if __name__ == "__main__":
    generate_full_report()