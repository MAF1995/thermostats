from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import requests


class MeteoClient:
    """Small helper around the Open-Meteo France endpoint."""

    BASE_URL = "https://api.open-meteo.com/v1/meteofrance"

    def __init__(self, latitude: float, longitude: float):
        self.lat = latitude
        self.lon = longitude

    def fetch_hourly(
        self,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        past_days: int = 2,
        forecast_days: int = 5,
    ) -> pd.DataFrame:
        """Download hourly data and return a normalized DataFrame."""

        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "hourly": "temperature_2m,windspeed_10m,relativehumidity_2m,shortwave_radiation",
            "timezone": "Europe/Paris",
        }
        if start_date and end_date:
            params["start_date"] = start_date.isoformat()
            params["end_date"] = end_date.isoformat()
        else:
            # `past_days` + `forecast_days` keeps the payload small while covering a week window.
            params["past_days"] = max(0, past_days)
            params["forecast_days"] = max(1, forecast_days)

        resp = requests.get(self.BASE_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly", {})

        df = pd.DataFrame(
            {
                "time": hourly.get("time", []),
                "temp_ext": hourly.get("temperature_2m", []),
                "wind": hourly.get("windspeed_10m", []),
                "humidity": hourly.get("relativehumidity_2m"),
                "solar_radiation": hourly.get("shortwave_radiation"),
            }
        )

        if df.empty:
            return df

        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).reset_index(drop=True)

        # Effective temperature takes into account wind cooling.
        temp_ext = pd.to_numeric(df["temp_ext"], errors="coerce")
        wind = pd.to_numeric(df["wind"], errors="coerce").fillna(0)
        df["temp_ext"] = temp_ext
        df["wind"] = wind
        df["temp_eff"] = temp_ext - 0.2 * wind

        return df

    def save_csv(self, output_path: str | Path, **kwargs) -> Path:
        """Fetch data then persist it locally for offline notebooks."""

        df = self.fetch_hourly(**kwargs)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path


def download_default_dataset(
    *,
    latitude: float = 48.8566,
    longitude: float = 2.3522,
    output_path: str | Path = "data/data_meteo.csv",
    past_days: int = 2,
    forecast_days: int = 5,
) -> Path:
    """Helper used by docs/tests to cache a ready-to-use CSV file."""

    client = MeteoClient(latitude, longitude)
    return client.save_csv(
        output_path,
        past_days=past_days,
        forecast_days=forecast_days,
    )


def _parse_cli_args():
    parser = argparse.ArgumentParser(description="Download Meteo-France (Open-Meteo) data to CSV.")
    parser.add_argument("--lat", "--latitude", dest="latitude", type=float, default=48.8566)
    parser.add_argument("--lon", "--longitude", dest="longitude", type=float, default=2.3522)
    parser.add_argument("--output", default="data/data_meteo.csv", help="CSV path to write.")
    parser.add_argument("--past-days", dest="past_days", type=int, default=2)
    parser.add_argument("--forecast-days", dest="forecast_days", type=int, default=5)
    parser.add_argument(
        "--start",
        dest="start",
        type=str,
        default=None,
        help="Optional YYYY-MM-DD start date (requires --end).",
    )
    parser.add_argument(
        "--end",
        dest="end",
        type=str,
        default=None,
        help="Optional YYYY-MM-DD end date (requires --start).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None
    client = MeteoClient(args.latitude, args.longitude)
    path = client.save_csv(
        args.output,
        start_date=start,
        end_date=end,
        past_days=args.past_days,
        forecast_days=args.forecast_days,
    )
    print(f"Dataset written to {path}")
