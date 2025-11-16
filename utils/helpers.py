import pandas as pd
import numpy as np

def to_datetime(series):
    return pd.to_datetime(series, errors="coerce")

def ensure_hourly(df, time_col="time"):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).resample("1H").interpolate()
    return df.reset_index()

def normalize(series):
    arr = np.array(series, dtype=float)
    if arr.std() == 0:
        return arr
    return (arr - arr.mean()) / arr.std()

def smooth(series, window=3):
    return pd.Series(series).rolling(window=window, min_periods=1).mean()

def clip_series(series, low=None, high=None):
    return pd.Series(series).clip(lower=low, upper=high)

def to_float(x):
    try:
        return float(x)
    except:
        return None

def align_series(series1, series2):
    s1 = pd.Series(series1).reset_index(drop=True)
    s2 = pd.Series(series2).reset_index(drop=True)
    n = min(len(s1), len(s2))
    return s1.iloc[:n], s2.iloc[:n]

def drop_outliers(series, zmax=3):
    s = pd.Series(series)
    z = (s - s.mean()) / s.std() if s.std() > 0 else 0
    return s[(np.abs(z) <= zmax)]
