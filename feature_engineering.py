"""
Data sources:
    - Daily bars: Yahoo Finance
"""
import os
import datetime as dt

import pandas as pd
import numpy as np

def calculate_returns(df: pd.DataFrame) -> pd.Series:
    return df["Close"] / df["Close"].shift(1) - 1

def calculate_overnight(df: pd.DataFrame) -> pd.Series:
    return df["Open"] / df["Close"].shift(1) - 1

def calculate_intraday_move(df: pd.DataFrame, col1: str, col2: str) -> pd.Series:
    return df[col2] / df[col1] - 1

def moving_normalized_volume(df: pd.DataFrame, n: int) -> pd.Series:
    ma = df["Volume"].rolling(n).mean()
    mstd = df["Volume"].rolling(n).std(ddof=1)
    return (df["Volume"] - ma) / mstd

def calculate_rsi(df: pd.DataFrame) -> pd.Series:
    avg_up = pd.Series(index=df.index, dtype=np.float64)
    avg_down = pd.Series(index=df.index, dtype=np.float64)

    up = df["Return"].apply(lambda x: max(x, 0))
    down = df["Return"].apply(lambda x: max(-x, 0))

    avg_up.iloc[1] = up.iloc[1]
    avg_down.iloc[1] = down.iloc[1]

    for i in range(2, len(df.index)):
        avg_up.iloc[i] = (avg_up.iloc[i-1] * 13 + up.iloc[i]) / 14
        avg_down.iloc[i] = (avg_down.iloc[i-1] * 13 + down.iloc[i]) / 14

    return 1 - (1 / (1 + avg_up / avg_down))

def calculate_rel_diff_ma(df: pd.DataFrame, col: str, n: int) -> pd.Series:
    ma = df[col].rolling(n).mean()
    return df[col] / ma - 1

def calculate_realized_volatility(df: pd.DataFrame, n: int) -> pd.Series:
    vol = df["Return"].rolling(n).std(ddof=1)
    return np.sqrt(252) * vol

if __name__ == "__main__":
    df = pd.read_csv(os.path.join("data", "GSPC.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    df["Return"] = calculate_returns(df)
    df["Open-Close"] = calculate_intraday_move(df, "Open", "Close")
    df["Open-Low"] = calculate_intraday_move(df, "Open", "Low")
    df["Open-High"] = calculate_intraday_move(df, "Open", "High")
    df["Close-MA 20D"] = calculate_rel_diff_ma(df, "Close", 20)
    df["RSI 14D"] = calculate_rsi(df)
    df["Normalized Volume"] = moving_normalized_volume(df, 30)

    df["Realized Volatility 30D"] = calculate_realized_volatility(df, 30)

    vix_df = pd.read_csv(os.path.join("data", "VIX.csv"))
    vix_df["Date"] = pd.to_datetime(vix_df["Date"])
    vix_df = vix_df.set_index("Date")

    df["VIX"] = vix_df["Close"]
    df["VIX Move"] = calculate_returns(vix_df)
    df["VIX Open-Close"] = calculate_intraday_move(vix_df, "Open", "Close")
    
    df = df.drop(columns=["Open", "High", "Low", "Close", "Volume", "Adj Close"])
    df = df[df.index >= dt.datetime(1995, 1, 1)]
    df.to_csv(os.path.join("data", "features.csv"))
