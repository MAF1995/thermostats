import json
import io
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import requests


class MapEngine:
    """Gestion centralisée des calques, zooms et frames animées pour la carte France."""

    DATA_URL = "https://public.opendatasoft.com/api/records/1.0/download/?dataset=communes-france%40public&type=csv"
    CACHE_FILE = Path("data/communes_cache.csv")

    def __init__(self):
        self._communes = None

    def load_communes(self) -> pd.DataFrame:
        if self._communes is not None:
            return self._communes

        if self.CACHE_FILE.exists():
            df = pd.read_csv(self.CACHE_FILE)
        else:
            try:
                resp = requests.get(self.DATA_URL, timeout=30)
                resp.raise_for_status()
                df = pd.read_csv(io.StringIO(resp.text))
                df.rename(
                    columns={"nom": "city", "population": "pop", "latitude": "lat", "longitude": "lon"},
                    inplace=True,
                )
                df = df[["city", "lat", "lon", "pop"]]
                df.to_csv(self.CACHE_FILE, index=False)
            except Exception:
                df = pd.DataFrame(
                    [
                        {"city": "Paris", "lat": 48.85, "lon": 2.35, "pop": 2161000},
                        {"city": "Lyon", "lat": 45.75, "lon": 4.85, "pop": 516000},
                        {"city": "Marseille", "lat": 43.30, "lon": 5.37, "pop": 861000},
                        {"city": "Toulouse", "lat": 43.6, "lon": 1.44, "pop": 493000},
                    ]
                )

        df.dropna(subset=["lat", "lon"], inplace=True)
        df["pop"] = df["pop"].fillna(0)
        self._communes = df
        return df

    def filter_by_zoom(self, df: pd.DataFrame, zoom: float) -> pd.DataFrame:
        if zoom < 5:
            return df[df["pop"] > 80000]
        if zoom < 7:
            return df[df["pop"] > 10000]
        return df

    def build_layer(self, df: pd.DataFrame, layer: str, values: pd.Series, colorscale: str, unit: str):
        return go.Scattermapbox(
            lat=df["lat"],
            lon=df["lon"],
            mode="markers",
            marker=dict(size=9, color=values, colorscale=colorscale, showscale=True, opacity=0.75,
                        colorbar=dict(title=f"{layer} ({unit})")),
            hovertemplate=("<b>%{customdata[0]}</b><br>" f"{layer}: %{{marker.color:.1f}} {unit}<extra></extra>"),
            customdata=df[["city"]].values,
            name=layer,
        )

    def calc_timelapse_frames(self, base_df: pd.DataFrame, timeline: List[pd.Timestamp], layers: Dict[str, Dict]):
        frames = []
        for i, ts in enumerate(timeline):
            traces = []
            for name, meta in layers.items():
                values = base_df[f"{name}_{i}"] if f"{name}_{i}" in base_df else base_df[name]
                traces.append(self.build_layer(base_df, name, values, meta["scale"], meta["unit"]))
            frames.append(go.Frame(data=traces, name=str(ts)))
        return frames

    def base_figure(self, center_lat=46.5, center_lon=2.5, zoom=4):
        return go.Figure(
            layout=go.Layout(
                mapbox_style="carto-positron",
                mapbox_center={"lat": center_lat, "lon": center_lon},
                mapbox_zoom=zoom,
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                height=650,
            )
        )

    def add_timelapse_controls(self, fig: go.Figure):
        fig.update_layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "y": 0,
                    "x": 0.05,
                    "buttons": [
                        {
                            "label": "▶",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 120, "redraw": True}, "fromcurrent": True}],
                        },
                        {"label": "⏸", "method": "animate", "args": [[None], {"mode": "immediate"}]},
                    ],
                }
            ],
            sliders=[
                {
                    "steps": [
                        {"args": [[f"{i}",], {"frame": {"duration": 0, "redraw": True}}], "label": str(i), "method": "animate"}
                        for i in range(len(fig.frames) if fig.frames else 0)
                    ]
                }
            ],
        )
        return fig

    def pick_on_map(self, fig: go.Figure, lat: float, lon: float):
        fig.add_trace(
            go.Scattermapbox(
                lat=[lat],
                lon=[lon],
                mode="markers+text",
                marker=dict(size=14, color="#ff7f0e"),
                text=["Votre position"],
                textposition="top center",
                name="Sélection",
            )
        )
        return fig

