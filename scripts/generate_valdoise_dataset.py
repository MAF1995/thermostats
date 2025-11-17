"""Generate a synthetic pellet-usage dataset for Val-d'Oise using real hourly temperatures.

The script downloads hourly temperature and wind data from the Open-Meteo archive API
for a location in Val-d'Oise (Pontoise). It then derives daily stove duration,
pellet consumption, and cost estimates using the PelletEngine heuristics so that the
resulting dataset follows the causal chain:

Température extérieure -> Durée d'utilisation -> Consommation de pellets -> Coût.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math
from pathlib import Path
import random
import sys

import pandas as pd
import requests

from core.pellet_engine import PelletEngine

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
VAL_DOISE_LAT = 49.050  # approx. Pontoise
VAL_DOISE_LON = 2.100
OUTPUT_PATH = Path("data/valdoise_pellet_dataset.csv")


@dataclass
class Scenario:
    target_temp: float = 22.0
    pellet_price: float = 4.8  # €/bag
    stove_power_kw: float = 8.0
    stove_efficiency: float = 0.85
    desired_duration: float = 18.0  # hours per bag goal
    base_hours: float = 1.5  # baseline usage when soft weather
    temp_sensitivity: float = 0.9  # added hours per °C below target
    wind_penalty: float = 0.25  # additional hours when wind exceeds 15 km/h


def fetch_weather(start: date, end: date) -> pd.DataFrame:
    params = {
        "latitude": VAL_DOISE_LAT,
        "longitude": VAL_DOISE_LON,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": "temperature_2m,windspeed_10m",
        "timezone": "Europe/Paris",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json().get("hourly", {})
    if not payload:
        raise RuntimeError("No hourly data returned by Open-Meteo archive API")

    df = pd.DataFrame(
        {
            "time": payload["time"],
            "temp_ext": payload.get("temperature_2m"),
            "wind": payload.get("windspeed_10m"),
        }
    )
    df["time"] = pd.to_datetime(df["time"], utc=False)
    df["date"] = df["time"].dt.date
    return df.dropna(subset=["temp_ext", "wind"]).reset_index(drop=True)


def derive_daily_metrics(df: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    pellet_engine = PelletEngine(
        power_kw=scenario.stove_power_kw,
        efficiency=scenario.stove_efficiency,
        pellet_price_bag=scenario.pellet_price,
    )

    grouped = df.groupby("date").agg(
        temp_avg=("temp_ext", "mean"),
        temp_min=("temp_ext", "min"),
        temp_max=("temp_ext", "max"),
        wind_avg=("wind", "mean"),
    )

    rows = []
    rng = random.Random(42)
    for day, stats in grouped.iterrows():
        delta = max(0.0, scenario.target_temp - stats.temp_avg)
        hours = scenario.base_hours + scenario.temp_sensitivity * delta
        if stats.wind_avg > 15:
            hours += scenario.wind_penalty * ((stats.wind_avg - 15) / 5)
        hours = max(0.0, min(24.0, hours))
        hours += rng.uniform(-0.5, 0.5)  # small behavioural noise
        hours = max(0.0, min(24.0, hours))

        effective_hours_for_rate = max(1, int(round(hours)))
        hourly_rate = pellet_engine.hourly_bag_rate(
            target_temp=scenario.target_temp,
            hours_on=effective_hours_for_rate,
            desired_duration_hours=scenario.desired_duration,
        )
        pellets_bags = hourly_rate * hours
        pellets_cost = pellets_bags * scenario.pellet_price

        rows.append(
            {
                "date": day,
                "temp_avg_c": round(stats.temp_avg, 2),
                "temp_min_c": round(stats.temp_min, 2),
                "temp_max_c": round(stats.temp_max, 2),
                "wind_avg_kmh": round(stats.wind_avg, 2),
                "duration_hours": round(hours, 2),
                "pellet_bags": round(pellets_bags, 3),
                "pellet_cost_eur": round(pellets_cost, 2),
            }
        )

    return pd.DataFrame(rows)


def main(days: int = 60):
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days)
    print(f"Fetching weather for Val-d'Oise from {start} to {end}...")
    hourly = fetch_weather(start, end)
    scenario = Scenario()
    daily = derive_daily_metrics(hourly, scenario)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset written to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    days = 60
    if len(sys.argv) > 1:
        try:
            days = max(7, int(sys.argv[1]))
        except ValueError:
            pass
    main(days)
