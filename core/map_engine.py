import json
import io
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests


class MapEngine:
    """Gestion centralisée des calques, zooms et frames animées pour la carte France."""

    DATA_URL = "https://public.opendatasoft.com/api/records/1.0/download/?dataset=communes-france%40public&type=csv"
    CACHE_FILE = Path("data/communes_cache.csv")

    def __init__(self):
        self._communes = None

    def _region_definitions(self):
        """Definitions approximatives des régions métropolitaines et de leurs départements."""

        return [
            {
                "name": "Île-de-France",
                "lat_range": (48.0, 49.3),
                "lon_range": (1.0, 3.5),
                "departments": ["75", "77", "78", "91", "92", "93", "94", "95"],
            },
            {
                "name": "Hauts-de-France",
                "lat_range": (49.7, 51.1),
                "lon_range": (1.4, 4.3),
                "departments": ["02", "59", "60", "62", "80"],
            },
            {
                "name": "Normandie",
                "lat_range": (48.3, 49.9),
                "lon_range": (-1.8, 1.7),
                "departments": ["14", "27", "50", "61", "76"],
            },
            {
                "name": "Bretagne",
                "lat_range": (47.5, 48.8),
                "lon_range": (-4.9, -1.0),
                "departments": ["22", "29", "35", "56"],
            },
            {
                "name": "Pays de la Loire",
                "lat_range": (46.4, 48.6),
                "lon_range": (-2.5, -0.5),
                "departments": ["44", "49", "53", "72", "85"],
            },
            {
                "name": "Centre-Val de Loire",
                "lat_range": (46.5, 48.9),
                "lon_range": (0.5, 2.5),
                "departments": ["18", "28", "36", "37", "41", "45"],
            },
            {
                "name": "Grand Est",
                "lat_range": (47.4, 49.5),
                "lon_range": (3.0, 7.7),
                "departments": ["08", "10", "51", "52", "54", "55", "57", "67", "68", "88"],
            },
            {
                "name": "Bourgogne-Franche-Comté",
                "lat_range": (46.1, 48.2),
                "lon_range": (2.7, 6.5),
                "departments": ["21", "25", "39", "58", "70", "71", "89", "90"],
            },
            {
                "name": "Nouvelle-Aquitaine",
                "lat_range": (43.0, 47.4),
                "lon_range": (-1.8, 2.0),
                "departments": ["16", "17", "19", "23", "24", "33", "40", "47", "64", "79", "86", "87"],
            },
            {
                "name": "Occitanie",
                "lat_range": (42.3, 45.1),
                "lon_range": (1.0, 3.8),
                "departments": ["09", "11", "12", "30", "31", "32", "34", "46", "48", "65", "66", "81", "82"],
            },
            {
                "name": "Auvergne-Rhône-Alpes",
                "lat_range": (44.1, 46.8),
                "lon_range": (2.0, 6.0),
                "departments": ["01", "03", "07", "15", "26", "38", "42", "43", "63", "69", "73", "74"],
            },
            {
                "name": "Provence-Alpes-Côte d'Azur",
                "lat_range": (43.0, 44.9),
                "lon_range": (4.0, 7.6),
                "departments": ["04", "05", "06", "13", "83", "84"],
            },
            {
                "name": "Corse",
                "lat_range": (41.3, 42.8),
                "lon_range": (8.5, 9.5),
                "departments": ["2A", "2B"],
            },
        ]

    def _offline_communes(self) -> pd.DataFrame:
        """Génère un maillage dense de communes si l'open data est inaccessible."""

        rng = np.random.default_rng(42)
        rows = []
        for region in self._region_definitions():
            lat_min, lat_max = region["lat_range"]
            lon_min, lon_max = region["lon_range"]
            for dept in region["departments"]:
                # densité suffisante pour couvrir ~35k points sur 96 départements (~360 par département)
                for i in range(360):
                    lat = rng.uniform(lat_min, lat_max)
                    lon = rng.uniform(lon_min, lon_max)
                    pop = int(rng.integers(200, 250000))
                    rows.append(
                        {
                            "city": f"Commune {dept}-{i+1}",
                            "lat": lat,
                            "lon": lon,
                            "pop": pop,
                            "region": region["name"],
                            "department": dept,
                        }
                    )
        return pd.DataFrame(rows)

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
                if "region" not in df:
                    df["region"] = "France"
                if "department" not in df:
                    df["department"] = "N/A"
                df = df[["city", "lat", "lon", "pop", "region", "department"]]
                df.to_csv(self.CACHE_FILE, index=False)
            except Exception:
                df = self._offline_communes()
                df.to_csv(self.CACHE_FILE, index=False)

        df.dropna(subset=["lat", "lon"], inplace=True)
        df["pop"] = df["pop"].fillna(0)
        self._communes = df
        return df

    def filter_by_zoom(self, df: pd.DataFrame, zoom: float) -> pd.DataFrame:
        if zoom < 4.5:
            threshold = df["pop"].quantile(0.93)
            return df[df["pop"] >= threshold]
        if zoom < 6:
            threshold = df["pop"].quantile(0.75)
            return df[df["pop"] >= threshold]
        if zoom < 7.5:
            return df.nlargest(15000, "pop")
        return df

    def build_layer(
        self,
        df: pd.DataFrame,
        layer: str,
        values: pd.Series,
        colorscale: str,
        unit: str,
        sizes: Sequence | None = None,
        label_fields: Sequence[str] | None = None,
        show_scale: bool = True,
    ):
        if label_fields is None:
            label_fields = ["city"] if "city" in df.columns else list(df.columns[:1])
        custom = df[label_fields].values
        colorbar = None
        if show_scale:
            colorbar = dict(title=f"{layer} ({unit})", x=-0.07, len=0.4)
        return go.Scattermapbox(
            lat=df["lat"],
            lon=df["lon"],
            mode="markers",
            marker=dict(
                size=sizes if sizes is not None else 9,
                color=values,
                colorscale=colorscale,
                showscale=show_scale,
                opacity=0.75,
                colorbar=colorbar,
            ),
            hovertemplate=("<b>%{customdata[0]}</b><br>" f"{layer}: %{{marker.color:.1f}} {unit}<extra></extra>"),
            customdata=custom,
            name=layer,
        )

    def calc_timelapse_frames(
        self,
        base_df: pd.DataFrame,
        timeline: List[pd.Timestamp],
        layers: Dict[str, Dict],
        aggregates: Dict[str, Dict] | None = None,
    ):
        frames = []
        for i, ts in enumerate(timeline):
            traces = []
            for name, meta in layers.items():
                col = meta.get("col", name)
                col_name = f"{col}_{i}"
                values = base_df[col_name] if col_name in base_df else base_df[col]
                traces.append(self.build_layer(base_df, name, values, meta["scale"], meta["unit"]))
            if aggregates:
                for level_name, agg_meta in aggregates.items():
                    agg_df = agg_meta.get("df", pd.DataFrame())
                    label_field = agg_meta.get("label", "city")
                    if agg_df.empty:
                        continue
                    for name, meta in layers.items():
                        col = meta.get("col", name)
                        col_name = f"{col}_{i}"
                        values = agg_df[col_name] if col_name in agg_df else agg_df[col]
                        traces.append(
                            self.build_layer(
                                agg_df,
                                f"{name} ({level_name})",
                                values,
                                meta["scale"],
                                meta["unit"],
                                sizes=[14 if level_name == "Régions" else 11] * len(agg_df),
                                label_fields=[label_field],
                            )
                        )
            frames.append(go.Frame(data=traces, name=str(ts)))
        return frames

    def aggregate_layers(self, base_df: pd.DataFrame, layers: Dict[str, Dict], by: str) -> pd.DataFrame:
        if by not in base_df.columns:
            return pd.DataFrame()
        numeric_cols = [c for c in base_df.columns if any(c.startswith(meta.get("col", name)) for name, meta in layers.items())]
        keep_cols = [c for c in numeric_cols if c in base_df.columns]
        agg = base_df.groupby(by)[keep_cols + ["lat", "lon"]].mean(numeric_only=True).reset_index()
        return agg

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
                        {
                            "args": [[frame.name], {"frame": {"duration": 0, "redraw": True}}],
                            "label": frame.name,
                            "method": "animate",
                        }
                        for frame in (fig.frames or [])
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

