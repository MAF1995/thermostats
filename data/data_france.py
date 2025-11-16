import numpy as np
import pandas as pd
import requests

from core.map_engine import MapEngine


class FranceMeteo:
    """Collecte météo simplifiée et projection sur l'ensemble des communes françaises."""

    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1/meteofrance"
        self.map_engine = MapEngine()

    def _anchor_communes(self, communes: pd.DataFrame, top_n: int = 40) -> pd.DataFrame:
        df = communes.copy()
        df["pop"] = pd.to_numeric(df["pop"], errors="coerce").fillna(0)
        return df.sort_values("pop", ascending=False).head(top_n)

    def _nearest_anchor_index(self, communes: pd.DataFrame, anchors: pd.DataFrame) -> np.ndarray:
        """Renvoie l'indice de la ville de référence la plus proche pour chaque commune."""
        comm_coords = communes[["lat", "lon"]].to_numpy(dtype=float)
        anchor_coords = anchors[["lat", "lon"]].to_numpy(dtype=float)
        # Distances euclidiennes approximatives suffisantes pour le filtrage zoom/couleur
        diff_lat = comm_coords[:, None, 0] - anchor_coords[None, :, 0]
        diff_lon = comm_coords[:, None, 1] - anchor_coords[None, :, 1]
        d2 = diff_lat**2 + diff_lon**2
        return np.argmin(d2, axis=1)

    def fetch_anchor_weather(self, anchors: pd.DataFrame, frames: int = 8) -> dict:
        """Télécharge un échantillon météo sur quelques grandes villes (ancres) pour alimenter la carte.

        On récupère un horizon de 24h (8 pas de 3h) puis on projette sur toutes les communes.
        """

        temps, winds, hums, apparents = [], [], [], []
        timeline = None
        for _, row in anchors.iterrows():
            params = {
                "latitude": row["lat"],
                "longitude": row["lon"],
                "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m,apparent_temperature",
                "timezone": "Europe/Paris",
            }
            try:
                resp = requests.get(self.base_url, params=params, timeout=20)
                data = resp.json().get("hourly", {})
                if timeline is None:
                    timeline = pd.to_datetime(data.get("time", [])[:frames * 3:3])
                temps.append(np.array(data.get("temperature_2m", [])[: frames * 3:3]))
                winds.append(np.array(data.get("windspeed_10m", [])[: frames * 3:3]))
                hums.append(np.array(data.get("relativehumidity_2m", [])[: frames * 3:3]))
                apparents.append(np.array(data.get("apparent_temperature", [])[: frames * 3:3]))
            except Exception:
                # Fallback doux si l'API est indisponible
                if timeline is None:
                    timeline = pd.date_range(pd.Timestamp.now(), periods=frames, freq="3H")
                base = np.linspace(2, 12, frames)
                temps.append(base)
                winds.append(np.linspace(5, 25, frames))
                hums.append(np.linspace(60, 90, frames))
                apparents.append(base - 1.5)

        return {
            "timeline": timeline if timeline is not None else pd.date_range(pd.Timestamp.now(), periods=frames, freq="3H"),
            "temps": np.array(temps),
            "winds": np.array(winds),
            "hums": np.array(hums),
            "apparents": np.array(apparents),
            "anchors": anchors.reset_index(drop=True),
        }

    def project_to_communes(self, communes: pd.DataFrame, anchor_weather: dict) -> pd.DataFrame:
        """Projette la météo des villes de référence sur chaque commune."""

        projected = communes.copy()
        idx = self._nearest_anchor_index(projected, anchor_weather["anchors"])

        frames = anchor_weather["timeline"].shape[0]
        for i in range(frames):
            projected[f"temp_{i}"] = anchor_weather["temps"][idx, i]
            projected[f"wind_{i}"] = anchor_weather["winds"][idx, i]
            projected[f"humidity_{i}"] = anchor_weather["hums"][idx, i]
            projected[f"apparent_{i}"] = anchor_weather["apparents"][idx, i]
        projected["start_hint"] = np.where(projected["temp_0"] < 5, "Allumage immédiat conseillé", "Surveiller la baisse nocturne")
        return projected

    def fetch(self, communes: pd.DataFrame, frames: int = 8) -> dict:
        """Pipeline complet : sélection d'ancres, requêtes météo, projection.

        Retourne un dict avec un DataFrame enrichi et la timeline correspondante.
        """

        anchors = self._anchor_communes(communes)
        anchor_weather = self.fetch_anchor_weather(anchors, frames=frames)
        enriched = self.project_to_communes(communes, anchor_weather)
        return {"data": enriched, "timeline": anchor_weather["timeline"]}
