import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def show_data():
    df = pd.read_csv("data/results/sliding_window_results.csv")
    df["selected_assets"] = df["selected_assets"].apply(lambda x: np.array(eval(x)))

    asset_matrix = np.vstack(df["selected_assets"].values)
    plt.figure(figsize=(10, 5))
    plt.imshow(asset_matrix.T, cmap="Greys", aspect="auto")
    plt.colorbar(label="Selected (1=yes, 0=no)")
    plt.xlabel("Window Index")
    plt.ylabel("Asset Index")
    plt.title("Asset Selection Over Sliding Windows")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(df["window_end"], df["objective_value"], marker="o")
    plt.xlabel("Window End Index")
    plt.ylabel("Objective Function Value")
    plt.title("Objective Function Value per Sliding Window")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    df_prices = pd.read_csv("data/data_apple_cocacola_google.csv", sep=";", parse_dates=["Date"])
    df_prices = df_prices.sort_values("Date").set_index("Date")
    df_prices = df_prices.apply(lambda col: col.str.replace(",", ".")).astype(float)
    log_returns = np.log(df_prices / df_prices.shift(1)).dropna().values

    actual_returns = []
    future_horizon = 5

    for _, row in df.iterrows():
        end = int(row["window_end"])
        if end + future_horizon >= len(log_returns):
            actual_returns.append(np.nan)
            continue

        weights = np.array(row["selected_assets"])
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
        cum_return = np.sum(log_returns[end:end + future_horizon] @ weights)
        actual_returns.append(cum_return)

    df["future_cum_return"] = actual_returns

    plt.figure(figsize=(10, 4))
    plt.plot(df["window_end"], df["future_cum_return"], marker="o", color='green')
    plt.xlabel("Window End Index")
    plt.ylabel("Cumulative Return (Next 5 Days)")
    plt.title("Future Cumulative Return Based on Selected Portfolio")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
