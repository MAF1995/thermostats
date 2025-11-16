import requests
import pandas as pd

class FranceMeteo:
    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1/meteofrance"

        self.coords = {
            "Paris": (48.85, 2.35),
            "Marseille": (43.30, 5.37),
            "Lyon": (45.75, 4.85),
            "Toulouse": (43.60, 1.44),
            "Nice": (43.70, 7.27),
            "Nantes": (47.21, -1.55),
            "Strasbourg": (48.58, 7.75),
            "Bordeaux": (44.84, -0.58),
            "Lille": (50.63, 3.06),
            "Rennes": (48.11, -1.68)
        }

    def fetch(self):
        data = []
        for city, (lat, lon) in self.coords.items():
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,windspeed_10m",
                "timezone": "Europe/Paris"
            }
            r = requests.get(self.base_url, params=params).json()
            temp = r["current"]["temperature_2m"]
            wind = r["current"]["windspeed_10m"]
            data.append([city, lat, lon, temp, wind])

        df = pd.DataFrame(data, columns=["city", "lat", "lon", "temp", "wind"])
        return df
