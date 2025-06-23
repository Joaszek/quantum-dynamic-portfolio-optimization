import numpy as np

from src.data_loader import load_data

def calculate_logs():
    df = load_data()
    log_returns = np.log(df / df.shift(1)).dropna()

    return log_returns
