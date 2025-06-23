import pandas as pd

def load_data():

    df = pd.read_csv("data/data_apple_cocacola_google.csv", sep=";", parse_dates=["Date"])
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)
    df = df.apply(lambda col: col.str.replace(",", ".")).astype(float)

    return df