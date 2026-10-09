"""Load and align GB demand (NESO) and weather (Open-Meteo / ERA5) data.

All timestamps are converted to UTC at half-hourly resolution.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Open-Meteo location_id -> city (order of the coordinates in the download URL)
CITIES = {0: "london", 1: "birmingham", 2: "manchester", 3: "glasgow"}


def load_demand(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Return half-hourly national demand (ND) indexed by UTC start time.

    NESO settlement periods count half-hours from local (UK) midnight, so clock-change
    days have 46 or 50 periods. Converting local midnight to UTC and adding
    (period - 1) * 30 min gives the correct UTC start time on every day.
    """
    frames = []
    for f in sorted(data_dir.glob("demanddata*.csv")):
        df = pd.read_csv(f)
        if "FORECAST_ACTUAL_INDICATOR" in df.columns:  # current-year file also holds forecasts
            df = df[df["FORECAST_ACTUAL_INDICATOR"] == "A"]
        df = df[["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND",
                 "EMBEDDED_SOLAR_CAPACITY", "EMBEDDED_WIND_CAPACITY"]].copy()
        # date formats differ between yearly files (01-Jan-23, 01-JAN-2024, 2025-01-01)
        iso = df["SETTLEMENT_DATE"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}")
        df["date"] = pd.NaT
        df.loc[iso, "date"] = pd.to_datetime(df.loc[iso, "SETTLEMENT_DATE"], format="%Y-%m-%d")
        df.loc[~iso, "date"] = pd.to_datetime(df.loc[~iso, "SETTLEMENT_DATE"], format="%d-%b-%y"
                                              if df.loc[~iso, "SETTLEMENT_DATE"].str.len().max() == 9
                                              else "%d-%b-%Y")
        df["date"] = pd.to_datetime(df["date"])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    local_midnight = df["date"].dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT")
    df["time"] = (local_midnight.dt.tz_convert("UTC")
                  + pd.to_timedelta((df["SETTLEMENT_PERIOD"] - 1) * 30, unit="min"))
    df = (df.dropna(subset=["time"])
            .drop_duplicates("time", keep="last")
            .set_index("time").sort_index())
    df = df.rename(columns={"ND": "demand", "SETTLEMENT_PERIOD": "settlement_period",
                            "EMBEDDED_SOLAR_CAPACITY": "solar_capacity",
                            "EMBEDDED_WIND_CAPACITY": "wind_capacity"})
    return df[["demand", "settlement_period", "solar_capacity", "wind_capacity"]]


def load_weather(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Return hourly weather averaged over four GB cities, interpolated to 30 min (UTC)."""
    f = next(data_dir.glob("open-meteo*.csv"))
    lines = f.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("location_id,time"))
    df = pd.read_csv(f, skiprows=start)
    df.columns = ["location_id", "time", "temperature", "solar_radiation", "wind_speed"]
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize("UTC")

    # unweighted mean across the four cities: a rough proxy for GB-wide conditions
    wx = df.groupby("time")[["temperature", "solar_radiation", "wind_speed"]].mean()
    wx = wx.resample("30min").interpolate("time")
    return wx


def load_all(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = load_demand(data_dir).join(load_weather(data_dir), how="inner")
    return df.dropna()


if __name__ == "__main__":
    df = load_all()
    print(df.head(), df.tail(), df.describe().T, sep="\n\n")
    print("\nrows:", len(df), "| span:", df.index.min(), "->", df.index.max())
    gaps = df.index.to_series().diff().value_counts().head()
    print("\nspacing counts:\n", gaps)
