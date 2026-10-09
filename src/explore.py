"""Exploratory plots: how GB demand depends on temperature, time of day and sunshine."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from data import load_all
from style import BLUE, ORANGE, INK, setup

OUT = Path(__file__).resolve().parents[1] / "results"


def main():
    setup()
    OUT.mkdir(exist_ok=True)
    df = load_all()
    local = df.tz_convert("Europe/London")

    # 1. Daily mean demand against daily mean temperature (weekdays only)
    daily = local.resample("D").mean(numeric_only=True)
    daily = daily[daily.index.dayofweek < 5]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(daily["temperature"], daily["demand"] / 1000, s=9, color=BLUE, alpha=0.5, lw=0)
    ax.set_xlabel("Daily mean temperature, 4-city average (°C)")
    ax.set_ylabel("Daily mean national demand (GW)")
    ax.set_title("Demand rises steeply below ~15 °C (weekdays, 2023 to 2026)")
    fig.tight_layout()
    fig.savefig(OUT / "explore_temperature.png", dpi=150)

    # 2. Average weekday profile in winter vs summer: midday dip from embedded solar
    wk = local[local.index.dayofweek < 5].copy()
    wk["hour"] = wk.index.hour + wk.index.minute / 60
    winter = wk[wk.index.month.isin([12, 1, 2])].groupby("hour")["demand"].mean() / 1000
    summer = wk[wk.index.month.isin([6, 7, 8])].groupby("hour")["demand"].mean() / 1000
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(winter.index, winter, color=BLUE, lw=2)
    ax.plot(summer.index, summer, color=ORANGE, lw=2)
    ax.legend(["Winter (Dec to Feb)", "Summer (Jun to Aug)"], frameon=False, loc="upper left")
    ax.set_xlabel("Local time of day (h)")
    ax.set_ylabel("Mean national demand (GW)")
    ax.set_xticks(range(0, 25, 3))
    ax.set_title("Average weekday profile: midday dip in summer from rooftop solar")
    fig.tight_layout()
    fig.savefig(OUT / "explore_daily_profile.png", dpi=150)

    # 3. Summer middays: demand falls as sunshine rises
    mid = local[(local.index.month.isin([5, 6, 7, 8])) & (local.index.hour.isin([12, 13]))
                & (local.index.dayofweek < 5)]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(mid["solar_radiation"], mid["demand"] / 1000, s=6, color=ORANGE, alpha=0.4, lw=0)
    ax.set_xlabel("Solar radiation, 4-city average (W/m²)")
    ax.set_ylabel("National demand (GW)")
    ax.set_title("Summer weekday middays: sunnier means lower demand on the grid")
    fig.tight_layout()
    fig.savefig(OUT / "explore_solar.png", dpi=150)

    print(f"weekday daily corr(temp, demand): {daily[['temperature', 'demand']].corr().iloc[0, 1]:.2f}")
    print(f"summer midday corr(radiation, demand): {mid[['solar_radiation', 'demand']].corr().iloc[0, 1]:.2f}")
    print("winter peak GW", winter.max().round(1), "| summer min GW", summer.min().round(1))


if __name__ == "__main__":
    main()
