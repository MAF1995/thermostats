import requests
import pandas as pd

class MeteoClient:
    def __init__(self, latitude, longitude):
        self.lat = latitude
        self.lon = longitude
        self.base_url = "https://api.open-meteo.com/v1/meteofrance"

    def fetch_hourly(self):
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "hourly": "temperature_2m,windspeed_10m",
            "timezone": "Europe/Paris"
        }

        r = requests.get(self.base_url, params=params)
        data = r.json()

        df = pd.DataFrame({
            "time": data["hourly"]["time"],
            "temp_ext": data["hourly"]["temperature_2m"],
            "wind": data["hourly"]["windspeed_10m"]
        })

        df["time"] = pd.to_datetime(df["time"])

        df["temp_eff"] = df["temp_ext"] - 0.2 * df["wind"]

        return df
