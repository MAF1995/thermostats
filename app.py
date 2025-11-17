import math
import html
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core.user_inputs import UserConfig, STRUCTURE_PRESETS, ISOLATION_NORM_METADATA
from data.data_meteo import MeteoClient
from models.thermal_model import ThermalModel
from core.kpi_engine import KPIEngine
from core.diagnostic import Diagnostic
from core.map_engine import MapEngine
from core.pellet_engine import PelletEngine
from data.data_france import FranceMeteo


DEFAULT_BAG_DURATION_HOURS = 18.0

STOVE_MODES = {
    "Confort": {
        "max_sessions": 3,
        "default_total_hours": 6,
        "max_session_hours": 3,
        "cooldown_minutes": 45,
        "strategy": "reactive",
        "anchor_hours": [6, 13, 20],
        "anchor_slack": 2,
        "description": "Jusqu'à 3 allumages pour rester proche de la consigne toute la journée.",
    },
    "Eco": {
        "max_sessions": 2,
        "default_total_hours": 4,
        "max_session_hours": 3,
        "cooldown_minutes": 30,
        "strategy": "reactive",
        "anchor_hours": [7, 19],
        "anchor_slack": 2,
        "description": "2 cycles maximum, priorité à la sobriété tout en gardant ±4°C.",
    },
    "Vacances": {
        "max_sessions": 1,
        "default_total_hours": 1,
        "max_session_hours": 1,
        "cooldown_minutes": 20,
        "strategy": "pulse",
        "floor_temp": 16.0,
        "anchor_hours": [14],
        "anchor_slack": 3,
        "description": "1 impulsion automatique pour éviter de passer sous 16°C.",
    },
}


HOUSING_DISTRIBUTION = pd.DataFrame(
    [
        {"Classe": "faible", "Taille": "<60 m²", "Logements": 820_000},
        {"Classe": "faible", "Taille": "60-120 m²", "Logements": 1_150_000},
        {"Classe": "faible", "Taille": "120-200 m²", "Logements": 740_000},
        {"Classe": "faible", "Taille": ">200 m²", "Logements": 210_000},
        {"Classe": "moyenne", "Taille": "<60 m²", "Logements": 540_000},
        {"Classe": "moyenne", "Taille": "60-120 m²", "Logements": 1_420_000},
        {"Classe": "moyenne", "Taille": "120-200 m²", "Logements": 960_000},
        {"Classe": "moyenne", "Taille": ">200 m²", "Logements": 260_000},
        {"Classe": "forte", "Taille": "<60 m²", "Logements": 310_000},
        {"Classe": "forte", "Taille": "60-120 m²", "Logements": 780_000},
        {"Classe": "forte", "Taille": "120-200 m²", "Logements": 630_000},
        {"Classe": "forte", "Taille": ">200 m²", "Logements": 180_000},
    ]
)


def smooth_hourly_columns(df: pd.DataFrame, columns):
    if df is None or df.empty:
        return df
    smoothed = df.copy()
    for base_col in columns:
        for idx in range(24):
            col_name = f"{base_col}_{idx}"
            if col_name not in smoothed.columns:
                continue
            neighbors = [smoothed[col_name]]
            prev_name = f"{base_col}_{idx-1}"
            next_name = f"{base_col}_{idx+1}"
            if prev_name in smoothed.columns:
                neighbors.append(smoothed[prev_name])
            if next_name in smoothed.columns:
                neighbors.append(smoothed[next_name])
            if len(neighbors) > 1:
                smoothed[col_name] = sum(neighbors) / len(neighbors)
    return smoothed


def categorize_home_size(volume_m3: float) -> str:
    if volume_m3 is None or volume_m3 <= 0:
        return "<60 m²"
    ceiling_height = 2.5  # approx. m
    area = volume_m3 / ceiling_height
    if area < 60:
        return "<60 m²"
    if area < 120:
        return "60-120 m²"
    if area < 200:
        return "120-200 m²"
    return ">200 m²"


def format_housing_count(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f} M"
    if count >= 10_000:
        return f"{round(count / 1_000):.0f} k"
    return f"{count:,}".replace(",", " ")


def build_isolation_legend_html(selected_code: str | None = None):
    rows = []
    for code, meta in ISOLATION_NORM_METADATA.items():
        tooltip = html.escape(meta.get("tooltip", ""))
        years = html.escape(meta.get("years", ""))
        label = html.escape(meta.get("label", ""))
        row_class = "iso-legend-row"
        if selected_code and code == selected_code:
            row_class += " iso-selected"
        rows.append(
            "".join(
                [
                    f"<div class='{row_class}' data-code='{code}'>",
                    f"<abbr title=\"{tooltip}\">{code}</abbr>",
                    f"<span class='iso-years'>{years}</span>",
                    f"<small>{label}</small>",
                    "</div>",
                ]
            )
        )
    rows_html = "\n".join(rows)
    style = (
        "<style>"
        ".iso-legend {border:1px solid #d7d7d7;border-radius:6px;padding:8px;margin-top:6px;font-size:0.78rem;max-height:220px;overflow-y:auto;}"
        ".iso-legend-row {display:flex;flex-direction:column;margin-bottom:6px;padding:4px;border-radius:4px;transition:background-color .2s ease;}"
        ".iso-legend-row:last-child {margin-bottom:0;}"
        ".iso-legend-row abbr {font-weight:600;cursor:help;text-decoration:none;border-bottom:1px dotted #999;}"
        ".iso-legend-row .iso-years {font-size:0.7rem;color:#6c757d;}"
        ".iso-legend-row small {color:#444;}"
        ".iso-legend-row.iso-selected {background-color:#fef5e7;border:1px solid #f5cba7;}"
        "</style>"
    )
    return style + f"<div class='iso-legend'>{rows_html}</div>"


def apply_timeline_styling_day_night(fig: go.Figure, hours):
    if hours is None:
        hours_list = []
    else:
        hours_list = list(hours)
    if not hours_list:
        return fig

    shapes = []
    night_blocks = [(0, 6), (20, 24)]
    total = len(hours_list)
    for start, end in night_blocks:
        if start >= total:
            continue
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=hours_list[start],
                x1=hours_list[min(end, total - 1)],
                y0=0,
                y1=1,
                fillcolor="rgba(20,20,60,0.08)",
                line=dict(width=0),
                layer="below",
            )
        )

    current_label = _resolve_current_label(hours_list) or hours_list[-1]

    shapes.append(
        dict(
            type="line",
            xref="x",
            yref="paper",
            x0=current_label,
            x1=current_label,
            y0=0,
            y1=1,
            line=dict(color="#2c3e50", width=2, dash="dot"),
        )
    )
    fig.update_layout(shapes=shapes)
    return fig


def _resolve_current_label(hours_list):
    if hours_list is None:
        values = []
    elif isinstance(hours_list, (pd.Series, pd.Index)):
        values = hours_list.tolist()
    else:
        try:
            values = list(hours_list)
        except TypeError:
            values = [hours_list]
    if not values:
        return None
    try:
        parsed = pd.to_datetime(values)
        tz = getattr(parsed, "tz", None)
        now = pd.Timestamp.now(tz=tz) if tz else pd.Timestamp.now()
        diffs = (parsed - now).to_series().abs()
        idx = int(diffs.idxmin())
        return parsed[idx]
    except Exception:
        idx = min(pd.Timestamp.now().hour, len(values) - 1)
        return pd.to_datetime(values[idx])


def _build_heating_plan(
    cfg: UserConfig,
    df: pd.DataFrame,
    model: ThermalModel,
    target_temp: float,
    hours_on: float,
    stove_mode: str,
):
    horizon = min(24, len(df))
    if horizon == 0:
        return {
            "horizon": 0,
            "heating_windows": [],
            "active_mask": [False] * 24,
            "active_hours": 0,
            "timestamps": pd.Series([], dtype="datetime64[ns]"),
            "hours_axis": [],
            "indoor_series": pd.Series(dtype="float64"),
            "wind_series": pd.Series(dtype="float64"),
            "temp_ext_series": pd.Series(dtype="float64"),
            "humidity_series": None,
            "radiation_series": None,
            "comfort_lower": target_temp - 4,
            "comfort_upper": target_temp + 4,
            "preheat_hours": 0,
            "cooldown_minutes": 0,
            "mode_label": stove_mode,
            "planned_sessions": 0,
            "session_hours": 0,
        }

    timestamps = df["time"].head(horizon).reset_index(drop=True)
    hours_axis = timestamps.tolist()
    wind_series = df["wind"].head(horizon).astype(float).reset_index(drop=True)
    temp_ext_series = df["temp_ext"].head(horizon).astype(float).reset_index(drop=True)
    humidity_series = (
        df["humidity"].head(horizon).astype(float).reset_index(drop=True) if "humidity" in df else None
    )
    radiation_col = next(
        (c for c in [
            "radiation",
            "solar",
            "solar_radiation",
            "shortwave_radiation",
            "global_radiation",
        ] if c in df.columns),
        None,
    )
    radiation_series = (
        df[radiation_col].head(horizon).astype(float).reset_index(drop=True) if radiation_col else None
    )

    mode_profile = STOVE_MODES.get(stove_mode, STOVE_MODES["Confort"])
    total_hours = float(hours_on) if hours_on else mode_profile["default_total_hours"]
    if mode_profile["strategy"] == "pulse":
        total_hours = mode_profile["default_total_hours"]
    total_hours = max(1.0, min(24.0, total_hours))
    max_sessions = max(1, mode_profile["max_sessions"])
    anchor_hours = mode_profile.get("anchor_hours") or []
    target_sessions = min(max_sessions, len(anchor_hours)) if anchor_hours else max_sessions
    session_hours = max(1, min(mode_profile["max_session_hours"], int(math.ceil(total_hours / target_sessions))))
    preheat_hours = max(1, min(6, int(round(cfg.tau_hours() / 2))))
    cooldown_minutes = mode_profile.get("cooldown_minutes", 30)
    cooldown_steps = max(1, math.ceil(cooldown_minutes / 60))

    lower_guard = target_temp - 4.0
    if mode_profile.get("floor_temp") is not None:
        lower_guard = max(lower_guard, mode_profile["floor_temp"])
    comfort_upper = target_temp + 4.0

    def simulate_mask(mask, clamp=False):
        series = model.simulate(
            float(cfg.temp_current),
            temp_ext_series.values,
            wind_series.values,
            hours_on=0,
            humidity_series=humidity_series.values if humidity_series is not None else None,
            target_temp=target_temp if clamp else None,
            custom_mask=mask,
        )
        if isinstance(series, pd.DataFrame):
            base_series = series.iloc[:, 0]
        elif isinstance(series, pd.Series):
            base_series = series
        else:
            base_series = pd.Series(series)
        out = base_series.reindex(range(horizon)).astype(float)
        if not out.empty:
            out.iloc[0] = float(cfg.temp_current)
        return out

    active_mask = [False] * horizon
    heating_windows = []
    planned_sessions = 0
    last_window_end = -cooldown_steps - 2

    temp_series = simulate_mask(active_mask, clamp=False)

    # reuse anchor metadata for scheduling below
    anchor_slack = mode_profile.get("anchor_slack", 2)

    def anchor_index(anchor_hour: float) -> int:
        if timestamps.empty:
            return 0
        base_day = timestamps.iloc[0].floor("D")
        anchor_ts = base_day + pd.Timedelta(hours=float(anchor_hour))
        if anchor_ts < timestamps.iloc[0]:
            anchor_ts += pd.Timedelta(days=1)
        diffs = (timestamps - anchor_ts).abs()
        idx = int(diffs.idxmin()) if not diffs.empty else 0
        return max(0, min(horizon - 1, idx))

    def locate_run_start(anchor_hour: float) -> int:
        target_idx = anchor_index(anchor_hour)
        if temp_series.empty:
            return target_idx
        window_start = max(0, target_idx - anchor_slack)
        window_end = min(horizon - 1, target_idx + anchor_slack)
        neighborhood = temp_series.loc[window_start:window_end]
        hits = neighborhood[neighborhood < lower_guard]
        if not hits.empty:
            return int(hits.index[0])
        future_hits = temp_series.loc[target_idx:][temp_series.loc[target_idx:] < lower_guard]
        if not future_hits.empty:
            return int(future_hits.index[0])
        past_window = temp_series.loc[:target_idx]
        past_hits = past_window[past_window < lower_guard]
        if not past_hits.empty:
            return int(past_hits.index[-1])
        return target_idx

    def schedule_window(run_start_idx: int):
        nonlocal temp_series, last_window_end
        run_start = max(0, min(horizon - 1, run_start_idx))
        min_run_idx = last_window_end + preheat_hours + 1
        if min_run_idx > run_start:
            run_start = min_run_idx
        if run_start >= horizon:
            return False
        preheat_start = max(0, run_start - preheat_hours)
        if preheat_start <= last_window_end:
            shift = (last_window_end - preheat_start) + 1
            run_start = min(horizon - 1, run_start + shift)
            preheat_start = max(0, run_start - preheat_hours)
        run_end = min(horizon - 1, run_start + session_hours - 1)
        for idx in range(preheat_start, run_end + 1):
            active_mask[idx] = True
        for idx in range(run_end + 1, min(horizon, run_end + 1 + cooldown_steps)):
            active_mask[idx] = True
        heating_windows.append(
            {
                "preheat_start": preheat_start,
                "start_idx": run_start,
                "end_idx": run_end,
                "cooldown_minutes": cooldown_minutes,
            }
        )
        last_window_end = run_end
        temp_series = simulate_mask(active_mask, clamp=False)
        return True

    if mode_profile["strategy"] == "pulse":
        anchor_idx = locate_run_start(anchor_hours[0]) if anchor_hours else 0
        if temp_series.empty:
            start_idx = anchor_idx
        else:
            min_idx = int(temp_series.idxmin())
            anchor_temp = temp_series.iloc[anchor_idx] if anchor_idx < len(temp_series) else None
            min_temp = temp_series.iloc[min_idx]
            if anchor_temp is not None and anchor_temp <= min_temp + 0.3:
                start_idx = anchor_idx
            else:
                start_idx = min_idx
        if schedule_window(start_idx):
            planned_sessions = 1
    elif anchor_hours:
        for anchor_hour in anchor_hours[:max_sessions]:
            run_start_idx = locate_run_start(anchor_hour)
            if schedule_window(run_start_idx):
                planned_sessions += 1
    else:
        for _ in range(max_sessions):
            violations = temp_series[temp_series < lower_guard]
            if violations.empty:
                break
            run_start_idx = int(violations.index[0])
            if schedule_window(run_start_idx):
                planned_sessions += 1

    indoor_series = simulate_mask(active_mask, clamp=True).clip(lower=-50, upper=50)
    mask_padded = active_mask + [False] * (24 - len(active_mask))
    active_hours = sum(1 for flag in mask_padded[:horizon] if flag)

    return {
        "horizon": horizon,
        "heating_windows": heating_windows,
        "active_mask": mask_padded,
        "active_hours": active_hours,
        "timestamps": timestamps,
        "hours_axis": hours_axis,
        "indoor_series": indoor_series,
        "wind_series": wind_series,
        "temp_ext_series": temp_ext_series,
        "humidity_series": humidity_series,
        "radiation_series": radiation_series,
        "comfort_lower": lower_guard,
        "comfort_upper": comfort_upper,
        "preheat_hours": preheat_hours,
        "cooldown_minutes": cooldown_minutes,
        "mode_label": stove_mode,
        "planned_sessions": planned_sessions,
        "session_hours": session_hours,
    }


def sidebar_inputs():
    st.sidebar.markdown("## Paramètres")
    st.sidebar.info("Ajustez les paramètres de votre habitation pour simuler le comportement thermique.")
    volume = st.sidebar.number_input("Volume chauffé (m³)", min_value=20.0, max_value=1200.0, value=250.0)
    inertia = st.sidebar.selectbox("Inertie thermique (label)", ["faible", "moyenne", "forte"], index=2)

    structure_options = UserConfig.structures()
    default_structure = "maçonnerie + polystyrène"
    structure_index = structure_options.index(default_structure) if default_structure in structure_options else 0
    structure = st.sidebar.selectbox(
        "Structure / paroi",
        structure_options,
        index=structure_index,
        help="Familles structure/isolant générées automatiquement (C et G).",
    )

    glazing_options = UserConfig.glazing_choices()
    glazing_default = "double vitrage"
    glazing_index = glazing_options.index(glazing_default) if glazing_default in glazing_options else 0
    glazing = st.sidebar.selectbox("Type de vitrage", glazing_options, index=glazing_index)

    vmc_options = UserConfig.vmc_choices()
    vmc_default = "simple flux"
    vmc_index = vmc_options.index(vmc_default) if vmc_default in vmc_options else 0
    vmc = st.sidebar.selectbox("VMC", vmc_options, index=vmc_index, help="Impacte les infiltrations d'air")

    power = st.sidebar.number_input("Puissance poêle (kW)", min_value=1.0, max_value=20.0, value=8.0)
    efficiency = st.sidebar.slider("Rendement", min_value=0.5, max_value=1.0, value=0.85)
    temp_current = st.sidebar.number_input("Température intérieure actuelle", value=19.0)
    temp_target = st.sidebar.number_input("Température intérieure cible", value=22.0)

    st.sidebar.markdown("### Localisation")
    map_engine = MapEngine()
    communes = map_engine.load_communes()
    pick_mode = st.sidebar.checkbox("Sélection sur carte (pick on map)", value=False)
    latitude = st.sidebar.number_input("Latitude", value=48.8, format="%0.5f")
    longitude = st.sidebar.number_input("Longitude", value=2.3, format="%0.5f")
    if pick_mode:
        selection = st.sidebar.selectbox("Commune", communes["city"].head(500).sort_values())
        selected_row = communes[communes["city"] == selection].iloc[0]
        latitude, longitude = float(selected_row["lat"]), float(selected_row["lon"])
        st.sidebar.caption("Les coordonnées sont synchronisées avec la carte France.")

    st.sidebar.markdown("### Norme d'isolation")
    isolation_options = list(ISOLATION_NORM_METADATA.keys())
    default_isolation = "RT2012" if "RT2012" in isolation_options else isolation_options[0]
    isolation = st.sidebar.selectbox(
        "Norme ou label",
        options=isolation_options,
        index=isolation_options.index(default_isolation),
        format_func=lambda key: f"{key} — {ISOLATION_NORM_METADATA[key]['label']}",
        help="Survolez les codes ci-dessous pour afficher les exigences thermiques historiques.",
    )
    iso_meta = ISOLATION_NORM_METADATA[isolation]
    st.sidebar.caption(f"{iso_meta['years']} · {iso_meta['tooltip']}")
    st.sidebar.markdown(build_isolation_legend_html(selected_code=isolation), unsafe_allow_html=True)

    st.sidebar.markdown("### Poêle")
    pellet_price = st.sidebar.number_input("Prix du sac de pellets (15 kg)", min_value=3.0, max_value=20.0, value=4.5)
    electric_price = st.sidebar.number_input("Prix électricité (€/kWh)", min_value=0.05, max_value=1.0, value=0.18, step=0.01)
    hours_on = st.sidebar.slider("Durée de chauffe prévue (h)", 0, 24, 6)

    mode_names = list(STOVE_MODES.keys())
    stove_mode = st.sidebar.radio(
        "Mode d'utilisation du poêle",
        mode_names,
        index=0,
        help="Confort = jusqu'à 3 cycles, Eco = 2 cycles sobres, Vacances = impulsion quotidienne automatique.",
    )
    st.sidebar.caption(STOVE_MODES[stove_mode]["description"])

    cfg = UserConfig(
        volume_m3=volume,
        inertia_level=inertia,
        power_kw=power,
        efficiency=efficiency,
        temp_current=temp_current,
        temp_target=temp_target,
        latitude=latitude,
        longitude=longitude,
        isolation=isolation,
        structure=structure,
        vmc=vmc,
        pellet_price_bag=pellet_price,
        glazing=glazing,
    )

    return cfg, pellet_price, electric_price, hours_on, stove_mode, map_engine


def load_meteo(cfg):
    client = MeteoClient(cfg.latitude, cfg.longitude)
    df = client.fetch_hourly()
    return df



def compute_kpis(cfg, df, hours_on, electric_price, pellet_df):
    kpi = KPIEngine(electric_price)
    base_cost = kpi.daily_cost(pellet_df, hours_on, standby_watts=60)
    deg = kpi.degree_day_ratio(df)
    base_loss = kpi.thermal_loss(cfg.volume_m3, cfg.tau_hours(), loss_w_per_k=cfg.loss_w_per_k())

    wind_mean = float(df["wind"].head(24).mean())
    humidity_mean = float(df["humidity"].head(24).mean()) if "humidity" in df else 60.0
    temp_mean = float(df["temp_ext"].head(24).mean())
    humidity_term = max(0.0, (humidity_mean - 60.0) / 40.0)
    wind_term = max(0.0, (wind_mean - 5.0) / 20.0)
    cold_term = max(0.0, (18.0 - temp_mean) / 15.0)
    weather_factor = 1 + 0.15 * wind_term + 0.1 * humidity_term + 0.2 * cold_term

    cost = base_cost * weather_factor
    loss = base_loss * weather_factor
    pellet_cost = kpi.pellet_cost(pellet_df)
    electric_cost = kpi.stove_electric_cost(hours_on)
    meteo_stats = {
        "wind_mean": wind_mean,
        "humidity_mean": humidity_mean,
        "temp_mean": temp_mean,
        "deg_day": deg,
        "weather_factor": weather_factor,
    }
    return cost, deg, loss, pellet_cost, electric_cost, meteo_stats


def render_weather_tab(df, pellet_df, hours_on):
    st.subheader("Sources météo et projections associées")
    horizon = min(24, len(df))
    if horizon == 0:
        st.warning("Impossible d'afficher les données météo.")
        return

    hours_axis = df["time"].head(horizon)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    temp_ext_series = df["temp_ext"].head(horizon).astype(float)
    fig.add_trace(
        go.Scatter(
            x=hours_axis,
            y=temp_ext_series.values,
            name="Température extérieure (°C)",
            mode="lines+markers",
            line=dict(color="#1f77b4"),
            marker=dict(color="#1f77b4"),
        ),
        secondary_y=False,
    )

    wind_series = df["wind"].head(horizon).astype(float)
    fig.add_trace(
        go.Scatter(
            x=hours_axis,
            y=wind_series.values,
            name="Vent (km/h)",
            mode="lines",
            line=dict(color="#4fc3f7", width=2),
            hovertemplate="Heure: %{x|%H:%M}<br>Vent: %{y:.0f} km/h<extra></extra>",
        ),
        secondary_y=True,
    )

    if "humidity" in df:
        fig.add_trace(
            go.Scatter(
                x=hours_axis,
                y=df["humidity"].head(horizon).values,
                name="Humidité relative (%)",
                mode="lines",
                line=dict(color="#6a1b9a", dash="dashdot"),
            ),
            secondary_y=True,
        )

    if "solar_radiation" in df:
        solar_values = df["solar_radiation"].head(horizon).astype(float)
        max_val = float(solar_values.max() or 1.0)
        sizes = [8 + 18 * (val / max_val) for val in solar_values]
        colors = [f"rgba(253,216,53,{0.15 + 0.75 * (val / max_val)})" for val in solar_values]
        fig.add_trace(
            go.Scatter(
                x=hours_axis,
                y=solar_values.values,
                name="Rayonnement solaire (W/m²)",
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=colors,
                    line=dict(color="#f9a825", width=1),
                ),
                hovertemplate="Heure: %{x|%H:%M}<br>Rayonnement: %{y:.0f} W/m²<extra></extra>",
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title="Fresque météo (24h)",
        legend_title="Séries",
    )
    temp_min = float(temp_ext_series.min()) if not temp_ext_series.empty else -10.0
    temp_max = float(temp_ext_series.max()) if not temp_ext_series.empty else 30.0
    y_min = max(-50, temp_min - 1.5)
    y_max = min(50, temp_max + 1.5)
    if y_max - y_min < 6:
        center = (y_min + y_max) / 2
        y_min = max(-50, center - 3)
        y_max = min(50, center + 3)
    fig.update_yaxes(title_text="Température (°C)", secondary_y=False, range=[y_min, y_max])
    fig.update_yaxes(title_text="Vent / HR / Radiation", secondary_y=True)
    fig = apply_timeline_styling_day_night(fig, hours_axis)
    st.plotly_chart(fig, use_container_width=True)

    meteo_cols = ["time", "temp_ext", "wind"]
    if "humidity" in df:
        meteo_cols.append("humidity")
    if "solar_radiation" in df:
        meteo_cols.append("solar_radiation")
    meteo_table = df[meteo_cols].head(horizon).rename(
        columns={
            "time": "Horodatage",
            "temp_ext": "Température (°C)",
            "wind": "Vent (km/h)",
            "humidity": "Humidité (%)",
            "solar_radiation": "Rayonnement (W/m²)",
        }
    )
    st.dataframe(meteo_table, use_container_width=True, hide_index=True)
    st.caption("Source météo : Open-Meteo (modèle Météo-France) — https://open-meteo.com")

    if not pellet_df.empty:
        pellet_table = pellet_df[["heure", "bags_used", "bags_cum", "cost_cum"]].copy()
        pellet_table.rename(
            columns={
                "heure": "Heure",
                "bags_used": "Sac/h",
                "bags_cum": "Sacs cumulés",
                "cost_cum": "Coût cumulé (€)",
            },
            inplace=True,
        )
        st.dataframe(pellet_table, use_container_width=True, hide_index=True)
        st.caption("Source consommation : moteur PelletEngine (simulation interne)")


def render_thermal_tab(
    cfg,
    df,
    model,
    pellet_df,
    target_temp,
    heating_plan,
):
    st.write("Simulation thermique basée sur l'inertie de l'habitation et la météo horaire.")
    st.caption(f"Consigne de chauffe : {target_temp:.1f}°C (saisie : {cfg.temp_target:.1f}°C).")

    horizon = heating_plan["horizon"]
    if horizon == 0:
        st.warning("Données météo indisponibles pour la simulation thermique.")
        return
    timestamps = heating_plan["timestamps"]
    hours_axis = heating_plan["hours_axis"]
    indoor_series = heating_plan["indoor_series"]
    wind_series = heating_plan["wind_series"]
    temp_ext_series = heating_plan["temp_ext_series"]
    humidity_series = heating_plan["humidity_series"]
    radiation_series = heating_plan["radiation_series"]
    heating_windows = heating_plan["heating_windows"]
    comfort_lower = heating_plan["comfort_lower"]
    comfort_upper = heating_plan["comfort_upper"]
    preheat_hours = heating_plan["preheat_hours"]
    cooldown_minutes = heating_plan["cooldown_minutes"]

    pellet_series = (
        pellet_df["bags_used"].head(horizon).astype(float).reset_index(drop=True)
        if not pellet_df.empty
        else pd.Series([0.0] * horizon)
    )

    timeline_df = pd.DataFrame({
        "Horodatage": timestamps,
        "Heure": timestamps.dt.strftime("%H:%M"),
        "Température intérieure projetée": indoor_series.values,
        "Température extérieure": temp_ext_series.values,
        "Vent": wind_series.values,
        "Pellets sac/h": pellet_series.values,
    })
    if humidity_series is not None and not humidity_series.empty:
        timeline_df["Humidité relative"] = humidity_series.values
    if radiation_series is not None and not radiation_series.empty:
        timeline_df["Rayonnement solaire"] = radiation_series.values

    for col in ["Température intérieure projetée", "Température extérieure"]:
        if col in timeline_df:
            timeline_df[col] = timeline_df[col].clip(lower=-50, upper=50)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    custom_cols = []
    hover_template = "Heure: %{x|%H:%M}<br>Intérieure: %{y:.1f}°C"
    if "Vent" in timeline_df:
        custom_cols.append(timeline_df["Vent"].values)
        idx = len(custom_cols) - 1
        hover_template += f"<br>Vent: %{{customdata[{idx}]:.0f}} km/h"
    if "Humidité relative" in timeline_df:
        custom_cols.append(timeline_df["Humidité relative"].values)
        idx = len(custom_cols) - 1
        hover_template += f"<br>HR: %{{customdata[{idx}]:.0f}}%"
    custom_cols.append(timeline_df["Pellets sac/h"].values)
    idx = len(custom_cols) - 1
    hover_template += f"<br>Pellets: %{{customdata[{idx}]:.2f}} sac/h<extra></extra>"
    custom_data = list(zip(*custom_cols)) if custom_cols else None

    fig.add_trace(
        go.Scatter(
            x=timeline_df["Horodatage"],
            y=timeline_df["Température intérieure projetée"],
            name="Température intérieure (°C)",
            mode="lines+markers",
            line=dict(color="#8d6e63", width=3),
            marker=dict(color="#8d6e63"),
            hovertemplate=hover_template,
            customdata=custom_data,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=timeline_df["Horodatage"],
            y=[target_temp] * len(timeline_df),
            name="Température souhaitée ajustée (°C)",
            mode="lines",
            line=dict(dash="dash", color="#c49a6c", width=3),
            hovertemplate="Température souhaitée : %{y:.1f}°C<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=timeline_df["Horodatage"],
            y=timeline_df["Température extérieure"],
            name="Température extérieure (°C)",
            mode="lines+markers",
            line=dict(color="#1f77b4"),
            marker=dict(color="#1f77b4"),
            hovertemplate="Heure: %{x|%H:%M}<br>Extérieure: %{y:.1f}°C<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=timeline_df["Horodatage"],
            y=timeline_df["Vent"],
            name="Vent (km/h)",
            mode="lines",
            line=dict(color="#4fc3f7", width=2),
            hovertemplate="Heure: %{x|%H:%M}<br>Vent: %{y:.0f} km/h<extra></extra>",
        ),
        secondary_y=True,
    )

    if "Humidité relative" in timeline_df:
        fig.add_trace(
            go.Scatter(
                x=timeline_df["Horodatage"],
                y=timeline_df["Humidité relative"],
                name="Humidité relative (%)",
                mode="lines",
                line=dict(color="#6a1b9a", dash="dashdot"),
                hovertemplate="Heure: %{x|%H:%M}<br>HR: %{y:.0f}%<extra></extra>",
            ),
            secondary_y=True,
        )

    if "Rayonnement solaire" in timeline_df:
        solar_vals = timeline_df["Rayonnement solaire"].fillna(0).astype(float)
        solar_max = max(1.0, float(solar_vals.max()))
        sizes = [10 + 25 * (val / solar_max) for val in solar_vals]
        colors = [f"rgba(253,216,53,{0.1 + 0.9 * (val / solar_max)})" for val in solar_vals]
        fig.add_trace(
            go.Scatter(
                x=timeline_df["Horodatage"],
                y=solar_vals,
                name="Rayonnement solaire (W/m²)",
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=colors,
                    line=dict(color="#f9a825", width=1),
                ),
                hovertemplate="Heure: %{x|%H:%M}<br>Rayonnement: %{y:.0f} W/m²<extra></extra>",
            ),
            secondary_y=True,
        )

    recharge_hours = []
    if "recharge" in pellet_df:
        recharge_idx = [int(i) for i in pellet_df.index[pellet_df["recharge"]]]
        recharge_hours = [hours_axis[i] for i in recharge_idx if i < len(hours_axis)]
    if recharge_hours:
        fig.add_trace(
            go.Scatter(
                x=recharge_hours,
                y=[timeline_df["Température intérieure projetée"].min() - 0.5] * len(recharge_hours),
                mode="markers+text",
                text=["Recharge pellet"] * len(recharge_hours),
                textposition="top center",
                marker_symbol="triangle-up",
                marker_color="#e67e22",
                name="Points de recharge",
                hovertemplate="Recharge pellet<extra></extra>",
            ),
            secondary_y=False,
        )

    fig.update_layout(
        xaxis=dict(domain=[0, 0.95]),
        legend_title="Courbes",
        title="Fresque temporelle unifiée (24h)",
    )
    temp_baseline = pd.Series([target_temp] * len(timeline_df)) if len(timeline_df) else pd.Series(dtype=float)
    temp_min = min(
        series.min()
        for series in [timeline_df["Température intérieure projetée"], timeline_df["Température extérieure"], temp_baseline]
        if not series.empty
    )
    temp_max = max(
        series.max()
        for series in [timeline_df["Température intérieure projetée"], timeline_df["Température extérieure"], temp_baseline]
        if not series.empty
    )
    y_min = max(-50, temp_min - 1.5)
    y_max = min(50, temp_max + 1.5)
    if y_max - y_min < 6:
        center = (y_max + y_min) / 2
        y_min = max(-50, center - 3)
        y_max = min(50, center + 3)
    fig.update_yaxes(title_text="Température (°C)", secondary_y=False, range=[y_min, y_max])
    fig.update_yaxes(title_text="Vent / HR / Rayonnement", secondary_y=True)

    fig = apply_timeline_styling_day_night(fig, hours_axis)

    ext_eff_series = temp_ext_series - 0.2 * wind_series
    now_marker = _resolve_current_label(timestamps) or timestamps.iloc[0]
    if isinstance(timestamps, pd.Series):
        timeline_series = pd.to_datetime(timestamps.reset_index(drop=True))
    else:
        timeline_series = pd.to_datetime(pd.Series(timestamps))
    try:
        idx_diffs = (timeline_series - now_marker).abs()
        now_idx = int(idx_diffs.idxmin())
    except Exception:
        now_idx = 0
    ext_future = ext_eff_series.iloc[now_idx:].values if now_idx < len(ext_eff_series) else []
    heating_path = model.time_series_until_target(
        float(cfg.temp_current),
        target_temp,
        ext_future,
    )
    if heating_path:
        curve_y = [float(cfg.temp_current)] + heating_path
        curve_y = [min(target_temp, y) for y in curve_y]
        curve_x = [now_marker] + [now_marker + pd.Timedelta(hours=i + 1) for i in range(len(heating_path))]
        fig.add_trace(
            go.Scatter(
                x=curve_x,
                y=curve_y,
                name="Courbe idéale",
                mode="lines",
                line=dict(color="#7f8c8d", width=2, dash="dot"),
                hovertemplate="Heure: %{x|%H:%M}<br>Trajectoire: %{y:.1f}°C<extra></extra>",
            ),
            secondary_y=False,
        )
        reach_time = curve_x[-1]
        st.caption(f"Si le poêle démarre maintenant, la consigne serait atteinte vers {reach_time:%H:%M}.")

    for idx, window in enumerate(heating_windows):
        heat_end_idx = min(window["end_idx"], len(hours_axis) - 1)
        start_time = hours_axis[window["start_idx"]]
        end_time = hours_axis[heat_end_idx]
        preheat_time = hours_axis[window["preheat_start"]]
        cooldown_end = end_time + pd.Timedelta(minutes=window.get("cooldown_minutes", cooldown_minutes))

        if window["preheat_start"] < window["start_idx"]:
            fig.add_vrect(
                x0=preheat_time,
                x1=start_time,
                fillcolor="rgba(255,183,77,0.20)",
                line_width=0,
                layer="below",
                annotation_text="Pré-chauffe" if idx == 0 else None,
                annotation_position="top left",
            )

        fig.add_vrect(
            x0=start_time,
            x1=end_time,
            fillcolor="rgba(255,87,34,0.28)",
            line_width=0,
            layer="below",
            annotation_text="Chauffe nominale" if idx == 0 else None,
            annotation_position="top right",
        )

        fig.add_vrect(
            x0=end_time,
            x1=cooldown_end,
            fillcolor="rgba(76,175,80,0.20)",
            line_width=0,
            layer="below",
            annotation_text="Cooldown" if idx == 0 else None,
            annotation_position="bottom right",
        )

    st.plotly_chart(fig, use_container_width=True)

    if heating_windows:
        st.caption(
            f"Bande de confort anticipée : {comfort_lower:.1f}°C – {comfort_upper:.1f}°C. Délais de préchauffe calculés sur {preheat_hours} h."
        )
        table_rows = []
        for idx, window in enumerate(heating_windows, start=1):
            preheat_time = timestamps.iloc[window["preheat_start"]]
            start_time = timestamps.iloc[window["start_idx"]]
            end_time = timestamps.iloc[min(window["end_idx"], len(timestamps) - 1)]
            cooldown_end = end_time + pd.Timedelta(minutes=window.get("cooldown_minutes", cooldown_minutes))
            preheat_duration = max(0, window["start_idx"] - window["preheat_start"])
            nominal_duration = max(1, window["end_idx"] - window["start_idx"] + 1)
            eff_temp = ext_eff_series.iloc[min(window["start_idx"], len(ext_eff_series) - 1)]
            time_to_target = model.time_to_reach(indoor_series.iloc[window["start_idx"]], target_temp, eff_temp)
            table_rows.append(
                {
                    "Fenêtre": idx,
                    "Pré-chauffe": preheat_time.strftime("%H:%M"),
                    "Allumage": start_time.strftime("%H:%M"),
                    "Arrêt estimé": end_time.strftime("%H:%M"),
                    "Fin cooldown": cooldown_end.strftime("%H:%M"),
                    "Durée préchauffe (h)": preheat_duration,
                    "Durée nominale (h)": nominal_duration,
                    "Temps pour atteindre la consigne (h)": round(time_to_target, 2) if time_to_target else "-",
                }
            )
        st.subheader("Fenêtres de chauffe prévues")
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)
    else:
        st.info("Aucune plage de chauffe anticipée n'est requise sur les prochaines 24 h.")



def render_kpi_tab(cost, deg, loss, pellet_cost, electric_cost, meteo_stats):
    st.write("Indicateurs calculés automatiquement à partir des données horaires et des paramètres de l'habitation.")
    st.divider()
    st.subheader("Indicateurs clés")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Coût journalier estimé (€)",
            round(cost, 2),
            help="Consommation pellets + électricité du poêle (veille) valorisées avec les prix saisis.",
        )
    with col2:
        st.metric(
            "Degré-jours (°C·jour)",
            round(deg, 2) if deg is not None else "N/A",
            help="Somme horaire des écarts à 18°C rapportée à la journée : mesure la sévérité du froid.",
        )
    with col3:
        st.metric(
            "Déperdition thermique (W/K)",
            round(loss, 2),
            help="Puissance perdue pour chaque degré d'écart entre intérieur et extérieur.",
        )

    st.markdown(
        f"Coût pellet: **{pellet_cost:.2f} €** — Coût élec du poêle: **{electric_cost:.2f} €**",
        help="Coût énergique journalier = consommation pellets + électricité du poêle",
    )

    weather_factor = meteo_stats.get("weather_factor", 1.0)
    wind_mean = meteo_stats.get("wind_mean")
    humidity_mean = meteo_stats.get("humidity_mean")
    st.caption(
        f"Impact météo : +{(weather_factor - 1) * 100:.1f}% (vent moyen {wind_mean:.1f} km/h, humidité {humidity_mean:.0f}% )."
    )

    if loss < 80:
        color = "#5CB85C"
        txt = "faible"
    elif loss < 150:
        color = "#F0AD4E"
        txt = "moyenne"
    else:
        color = "#D9534F"
        txt = "forte"

    st.markdown(
        f"""
        <div style="padding: 12px; border-radius: 8px; background-color: {color}; color: white;">
            Niveau de pertes thermiques : <b>{txt}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_diagnostic_tab(cfg, loss, meteo_stats):
    st.write("Évaluation du comportement thermique de l'habitation selon les pertes estimées.")
    st.divider()
    st.subheader("Diagnostic thermique")

    diag = Diagnostic(loss, cfg, meteo_stats)
    result = diag.summary()

    cls = result["classe"]
    score = result["score"]
    exp = result["explication"]
    rec = result["recommandation"]
    construction = result["construction"]

    if cls == "faible":
        color = "#5CB85C"
    elif cls == "moyenne":
        color = "#F0AD4E"
    else:
        color = "#D9534F"

    st.markdown(
        f"""
        <div style="padding: 14px; border-radius: 6px; background-color: {color}; color: white; font-weight: bold;">
            Niveau de pertes thermiques : {cls.upper()}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.subheader("Indice de sévérité")
    pct = score / 3 * 100
    st.progress(pct / 100)

    st.write("")
    st.subheader("Explication")
    st.write(exp)

    st.write("")
    st.subheader("Recommandation")
    st.write(rec)

    st.subheader("Profil de construction")
    st.write(construction)

    st.subheader("Parc résidentiel comparable")
    size_label = categorize_home_size(cfg.volume_m3)
    housing_peer_df = HOUSING_DISTRIBUTION[HOUSING_DISTRIBUTION["Classe"] == cls]
    tranche = housing_peer_df[housing_peer_df["Taille"] == size_label]
    if not tranche.empty:
        count = int(tranche.iloc[0]["Logements"])
        st.metric("Logements similaires", format_housing_count(count), help="Nombre estimé d'habitations en France partageant taille et classe énergétique.")
    st.dataframe(
        housing_peer_df.rename(columns={"Taille": "Surface", "Logements": "Parc (logements)"}),
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("Détails du diagnostic"):
        st.markdown(
            "- **Isolation saisie** : {isolation}\n"
            "- **Pertes estimées** : {loss:.1f} W/K — plus la valeur est élevée, plus la maison se refroidit vite.\n"
            "- **Inertie thermique** : {inertia} (impacte la vitesse de montée et de descente en température).\n"
            "- **Conseil** : surveiller les infiltrations d'air, les vitrages et l'isolation des combles pour améliorer le score."
            .format(isolation=cfg.isolation, loss=loss, inertia=cfg.inertia_level)
        )

    st.subheader("Comparaison régionale")
    comparison_df, region_label = build_diagnostic_comparison(cfg, loss, meteo_stats)
    family_counts = HOUSING_DISTRIBUTION.groupby("Classe")[["Logements"]].sum().reset_index()
    family_counts["Pourcentage"] = (
        family_counts["Logements"] / family_counts["Logements"].sum() * 100
    )
    pie_colors = {"faible": "#d35400", "moyenne": "#f1c40f", "forte": "#27ae60"}
    pie_fig = go.Figure(
        go.Pie(
            labels=family_counts["Classe"],
            values=family_counts["Logements"],
            hole=0.35,
            direction="clockwise",
            marker=dict(colors=[pie_colors.get(lbl, "#95a5a6") for lbl in family_counts["Classe"]]),
            pull=[0.1 if lbl == cls else 0 for lbl in family_counts["Classe"]],
            hovertemplate="Famille %{label}<br>Part: %{percent:.1%}<br>Logements: %{value:,}<extra></extra>",
        )
    )
    pie_fig.update_layout(
        title=f"Répartition des familles de maisons (réf. nationale) — Vous: {cls}",
        legend_title="Famille",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(pie_fig, use_container_width=True)


def build_diagnostic_comparison(cfg, loss, meteo_stats):
    map_engine = MapEngine()
    region_label = "France"
    try:
        communes = map_engine.load_communes()
        deltas = (communes["lat"] - cfg.latitude) ** 2 + (communes["lon"] - cfg.longitude) ** 2
        idx = deltas.idxmin()
        region_label = communes.loc[idx].get("region", region_label)
    except Exception:
        pass

    region_adjust = {
        "Hauts-de-France": 1.12,
        "Bretagne": 0.98,
        "Île-de-France": 1.02,
        "Auvergne-Rhône-Alpes": 1.08,
        "Provence-Alpes-Côte d'Azur": 0.92,
        "Occitanie": 0.95,
    }.get(region_label, 1.0)
    regional_loss = round(loss * region_adjust, 1)
    national_loss = round(max(70.0, loss * 1.03), 1)

    wind = meteo_stats.get("wind_mean", 10.0)
    regional_wind = wind * (0.9 if region_adjust < 1 else 1.05)
    national_wind = 12.0

    deg_day = meteo_stats.get("deg_day", 0.0) or 0.0
    regional_deg = round(max(0.0, deg_day * region_adjust), 1)
    national_deg = round(max(10.0, deg_day * 1.05), 1)
    comparison_df = pd.DataFrame(
        [
            {
                "Profil": "Votre foyer",
                "Pertes W/K": round(loss, 1),
                "Vent km/h": round(wind, 1),
                "Degré-jours": round(deg_day, 1),
            },
            {
                "Profil": f"Moyenne {region_label}",
                "Pertes W/K": regional_loss,
                "Vent km/h": round(regional_wind, 1),
                "Degré-jours": regional_deg,
            },
            {
                "Profil": "Moyenne France",
                "Pertes W/K": national_loss,
                "Vent km/h": national_wind,
                "Degré-jours": national_deg,
            },
        ]
    )
    return comparison_df, region_label



def render_map_tab(cfg, map_engine: MapEngine):
    st.write("Carte interactive des températures en France.")
    st.divider()

    st.subheader("Carte thermique de la France")
    fm = FranceMeteo()

    zoom = st.slider("Zoom carte", min_value=3.0, max_value=8.0, value=4.5, step=0.5)
    communes_all = map_engine.load_communes()
    communes = map_engine.filter_by_zoom(communes_all, zoom)
    communes = communes.rename(columns={"nom": "city"}) if "nom" in communes.columns else communes

    weather = fm.fetch(communes, frames=24)
    df_fr = weather["data"]
    timeline = list(weather["timeline"].strftime("%d/%m %Hh"))

    layer_options = {
        "Température": ("temp", "RdBu_r", "°C"),
        "Humidité": ("humidity", "Blues", "%"),
        "Vent": ("wind", "PuBu", "km/h"),
        "Ressentie": ("apparent", "OrRd", "°C"),
        "Rayonnement": ("radiation", "YlOrBr", "W/m²"),
    }

    df_fr = smooth_hourly_columns(df_fr, [spec[0] for spec in layer_options.values()])

    selected_layers = st.multiselect(
        "Couches à afficher (gradients)",
        list(layer_options.keys()),
        default=["Température"],
    )
    if not selected_layers:
        selected_layers = ["Température"]

    base_fig = map_engine.base_figure(center_lat=cfg.latitude, center_lon=cfg.longitude, zoom=int(round(zoom)))

    size_hint = [11] * len(df_fr)

    layers_meta = {
        name: {"scale": layer_options[name][1], "unit": layer_options[name][2], "col": layer_options[name][0]}
        for name in selected_layers
    }

    def _pick_series(source: pd.DataFrame, base_col: str) -> pd.Series:
        """Sélectionne la première série disponible pour un calque (col_0, col ou préfixe)."""
        ordered_candidates = [f"{base_col}_0", base_col]
        ordered_candidates.extend(sorted(c for c in source.columns if c.startswith(f"{base_col}_")))
        for candidate in ordered_candidates:
            if candidate in source.columns:
                return source[candidate]
        # Retourne une série NaN pour conserver la taille attendue sans lever d'erreur.
        return pd.Series([float("nan")] * len(source), index=source.index)

    for idx, layer in enumerate(selected_layers):
        col, scale, unit = layer_options[layer]
        base_fig.add_trace(
            map_engine.build_layer(
                df_fr,
                layer,
                _pick_series(df_fr, col),
                scale,
                unit,
                sizes=size_hint,
                show_scale=idx == 0,
            )
        )

    frames = map_engine.calc_timelapse_frames(
        df_fr,
        timeline,
        layers_meta,
        aggregates=None,
    )
    base_fig.frames = frames
    map_engine.add_timelapse_controls(base_fig)
    map_engine.pick_on_map(base_fig, cfg.latitude, cfg.longitude)
    base_fig.add_annotation(
        x=1.03,
        y=0.5,
        xref="paper",
        yref="paper",
        text="Sources: Open-Meteo · INSEE",
        showarrow=False,
        align="left",
    )

    st.caption("Pastilles redimensionnées selon l'échelle de carte et colorées par couches météorologiques sélectionnées.")
    st.plotly_chart(base_fig, use_container_width=True)



def main():
    st.title("Thermo-Stats")

    cfg, pellet_price, electric_price, hours_on, stove_mode, map_engine = sidebar_inputs()

    if not cfg.validate():
        st.error("Paramètres invalides.")
        return
    st.markdown(
        """
        <div style="padding: 12px; background-color: #f0f2f6; border-radius: 8px; border: 1px solid #d0d0d0;">
            Analyse thermique, prévisions et recommandations pour l'habitation chauffée au poêle.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.divider()

    df = load_meteo(cfg)
    target_temp_effective = cfg.temp_target
    model = ThermalModel(
        tau_hours=cfg.tau_hours(),
        volume_m3=cfg.volume_m3,
        power_kw=cfg.power_kw,
        efficiency=cfg.efficiency,
        loss_w_per_k=cfg.loss_w_per_k(),
        capacitance_kwh=cfg.capacitance_kwh(),
    )
    heating_plan = _build_heating_plan(
        cfg,
        df,
        model,
        target_temp_effective,
        hours_on,
        stove_mode,
    )
    active_mask = heating_plan["active_mask"] if heating_plan["horizon"] > 0 else [i < hours_on for i in range(24)]
    effective_hours_on = heating_plan["active_hours"] if heating_plan["horizon"] > 0 else hours_on
    pellet_hours_for_rate = max(1, int(effective_hours_on))
    effective_hours_for_kpis = max(0, int(effective_hours_on))
    pellet_engine = PelletEngine(cfg.power_kw, cfg.efficiency, cfg.pellet_price_bag)
    pellet_df = pellet_engine.compute_pellet_usage(
        hours=24,
        target_temp=target_temp_effective,
        hours_on=pellet_hours_for_rate,
        desired_duration_hours=DEFAULT_BAG_DURATION_HOURS,
        active_mask=active_mask,
    )

    cost, deg, loss, pellet_cost, electric_cost, meteo_stats = compute_kpis(
        cfg,
        df,
        effective_hours_for_kpis,
        electric_price,
        pellet_df,
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Thermique", "Météo", "KPIs", "Diagnostic", "Carte France"])

    with tab1:
        render_thermal_tab(
            cfg,
            df,
            model,
            pellet_df,
            target_temp_effective,
            heating_plan,
        )

    with tab2:
        render_weather_tab(df, pellet_df, hours_on)

    with tab3:
        render_kpi_tab(cost, deg, loss, pellet_cost, electric_cost, meteo_stats)

    with tab4:
        render_diagnostic_tab(cfg, loss, meteo_stats)

    with tab5:
        render_map_tab(cfg, map_engine)

    st.markdown(
        """
        <hr>
        <div style="text-align: center; color: gray;">
            Thermo-Stats — Prototype énergétique © 2025
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
