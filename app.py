from __future__ import annotations

import math
import html
import os
import time
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
from core.pellet_engine import PelletEngine, KG_PER_BAG
from data.data_france import FranceMeteo


DEFAULT_BAG_DURATION_HOURS = 14.0
PLANNING_HORIZON_HOURS = 48
STANDARD_HOME_SETTINGS = {
    "area_m2": 100,
    "ceiling_height": 2.5,
    "tau_hours": 5.5,
    "power_kw": 7.5,
    "efficiency": 0.86,
    "max_delta_per_hour": 2.2,
}
STANDARD_HOME_MODEL = ThermalModel(
    tau_hours=STANDARD_HOME_SETTINGS["tau_hours"],
    volume_m3=STANDARD_HOME_SETTINGS["area_m2"] * STANDARD_HOME_SETTINGS["ceiling_height"],
    power_kw=STANDARD_HOME_SETTINGS["power_kw"],
    efficiency=STANDARD_HOME_SETTINGS["efficiency"],
)

STOVE_MODES = {
    "Confort": {
        "min_sessions": 3,
        "max_sessions": 4,
        "default_total_hours": 6,
        "max_session_hours": 3,
        "cooldown_minutes": 45,
        "strategy": "reactive",
        "anchor_hours": [6, 13, 20],
        "anchor_slack": 2,
        "max_delta_per_hour": 2.5,
        "description": "Entre 3 et 4 allumages resserrés pour rester proche de la consigne toute la journée.",
    },
    "Eco": {
        "max_sessions": 2,
        "default_total_hours": 4,
        "max_session_hours": 3,
        "cooldown_minutes": 30,
        "strategy": "reactive",
        "anchor_hours": [7, 19],
        "anchor_slack": 2,
        "max_delta_per_hour": 1.8,
        "description": "2 cycles maximum, priorité à la sobriété tout en gardant ±4°C.",
    },
    "Vacances": {
        "max_sessions": 1,
        "default_total_hours": 4,
        "max_session_hours": 6,
        "cooldown_minutes": 20,
        "strategy": "pulse",
        "floor_temp": 16.0,
        "restore_temp": 19.0,
        "pulse_target_temp": 20.0,
        "anchor_hours": [14],
        "anchor_slack": 3,
        "max_delta_per_hour": 1.0,
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

VMC_METADATA = {
    "aucune": {
        "label": "Ventilation naturelle",
        "tooltip": "Sans VMC, l'humidité reste prisonnière : l'air se réchauffe lentement car une partie de l'énergie sert à assécher les murs.",
        "annual_kwh": 0,
        "dryness_hint": "Air humide, chauffage ralenti",
    },
    "simple flux": {
        "label": "Simple flux permanent",
        "tooltip": "Une simple flux extrait l'air humide des pièces d'eau pour aider le poêle à chauffer un air déjà plus sec.",
        "annual_kwh": 110,
        "dryness_hint": "Air extrait en continu",
    },
    "hygro A": {
        "label": "Hygro A (bouches auto)",
        "tooltip": "Les bouches higro A s'ouvrent avec l'humidité : l'air sec accélère la montée à 22°C.",
        "annual_kwh": 150,
        "dryness_hint": "Débits modulés",
    },
    "hygro B": {
        "label": "Hygro B (entrées + bouches)",
        "tooltip": "Hygro B ajuste extraction et insufflation pour maintenir un air plus sec et limiter les pertes.",
        "annual_kwh": 180,
        "dryness_hint": "Air équilibré",
    },
    "double flux": {
        "label": "Double flux HR",
        "tooltip": "La double flux récupère la chaleur de l'air extrait, tout en séchant l'air neuf insufflé.",
        "annual_kwh": 260,
        "dryness_hint": "Récupération de chaleur",
    },
    "double flux thermodynamique": {
        "label": "Double flux thermo",
        "tooltip": "Le module thermodynamique préchauffe l'air entrant : plus sec et à bonne température pour soulager le poêle.",
        "annual_kwh": 320,
        "dryness_hint": "Air préchauffé",
    },
}

VMC_ANNUAL_KWH = {key: meta["annual_kwh"] for key, meta in VMC_METADATA.items()}


def _blend_hex(color_a: str, color_b: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, float(ratio)))
    try:
        ca = tuple(int(color_a[i:i + 2], 16) for i in (1, 3, 5))
        cb = tuple(int(color_b[i:i + 2], 16) for i in (1, 3, 5))
    except Exception:
        return color_a
    mixed = tuple(int(ca[idx] + (cb[idx] - ca[idx]) * ratio) for idx in range(3))
    return "#" + "".join(f"{channel:02X}" for channel in mixed)


def temperature_color(temp: float | None) -> str:
    if temp is None:
        return "#30D158"
    value = float(temp)
    if value <= 10:
        return "#3DC2FF"
    if value <= 19:
        return _blend_hex("#3DC2FF", "#30D158", (value - 10) / 9)
    if value <= 24:
        return _blend_hex("#30D158", "#7CF17F", (value - 19) / 5)
    if value <= 30:
        return _blend_hex("#7CF17F", "#FFB347", (value - 24) / 6)
    return "#FF8C42"


def performance_color(ratio: float | None) -> str:
    if ratio is None:
        return "#29D391"
    pct = max(0.0, min(1.0, float(ratio)))
    return _blend_hex("#FF5F6D", "#29D391", pct)


def poll_domotic_sensor(sensor_id: str | None) -> float | None:
    if not sensor_id:
        return None
    normalized = sensor_id.strip()
    if not normalized:
        return None
    sensor_cache = st.session_state.get("domotic_sensors")
    if isinstance(sensor_cache, dict) and normalized in sensor_cache:
        try:
            return float(sensor_cache[normalized])
        except (TypeError, ValueError):
            pass
    try:
        secrets_block = st.secrets.get("domotic_sensors", {})  # type: ignore[attr-defined]
        if normalized in secrets_block:
            return float(secrets_block[normalized])
    except Exception:
        pass
    env_key = f"DOMOTIC_SENSOR_{normalized.upper()}"
    env_value = os.environ.get(env_key)
    if env_value is not None:
        try:
            return float(env_value)
        except ValueError:
            return None
    return None


def _safe_rerun():
    rerun_fn = getattr(st, "rerun", None)
    if callable(rerun_fn):
        rerun_fn()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()  # type: ignore[attr-defined]


def render_circular_performance_chart(score: float | None):
    severity = 1.0 if score is None else max(1.0, min(3.0, float(score)))
    severity_ratio = max(0.0, min(1.0, (severity - 1.0) / 2.0))
    performance_ratio = 1.0 - severity_ratio
    performance_pct = performance_ratio * 100
    core_color = performance_color(performance_ratio)

    segments = 48
    theta_segment = [idx * (360 / segments) for idx in range(segments)]
    ring_colors = [performance_color(idx / (segments - 1)) for idx in range(segments)]

    fig = go.Figure()
    fig.add_trace(
        go.Barpolar(
            r=[1.05] * segments,
            theta=theta_segment,
            width=[360 / segments] * segments,
            opacity=0.85,
            marker=dict(color=ring_colors, line=dict(color="rgba(0,0,0,0)", width=0)),
            hoverinfo="skip",
        )
    )

    resolution = 120
    theta_fill = [idx * (360 / resolution) for idx in range(resolution + 1)]
    r_fill = [performance_ratio] * len(theta_fill)
    fig.add_trace(
        go.Scatterpolar(
            theta=theta_fill,
            r=r_fill,
            fill="toself",
            mode="lines",
            line=dict(color=core_color, width=2),
            fillcolor=_blend_hex("#FFFFFF", core_color, 0.25),
            hovertemplate=f"Performance : {performance_pct:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 1.1], visible=False),
            angularaxis=dict(showticklabels=False, ticks="", rotation=90, direction="clockwise"),
        ),
    )

    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text=(
            "<span style='font-size:15px;color:#A8ADB5;'>Performance</span><br>"
            f"<span style='font-size:30px;color:{core_color};font-weight:bold;'>{performance_pct:.0f}%</span>"
        ),
        showarrow=False,
        align="center",
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


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


def summarize_pellet_usage(pellet_df: pd.DataFrame, horizon_hours: int, hours_per_day: int = 24) -> dict:
    if pellet_df is None or pellet_df.empty:
        return {
            "total_bags": 0.0,
            "total_kg": 0.0,
            "per_day": [],
            "coverage_hours": max(0, horizon_hours),
        }
    coverage = min(max(0, horizon_hours), len(pellet_df))
    usage_slice = pellet_df["bags_used"].iloc[:coverage]
    total_bags = float(usage_slice.sum())
    per_day = []
    if hours_per_day > 0:
        for start in range(0, coverage, hours_per_day):
            per_day.append(float(usage_slice.iloc[start:start + hours_per_day].sum()))
    return {
        "total_bags": total_bags,
        "total_kg": total_bags * KG_PER_BAG,
        "per_day": per_day,
        "coverage_hours": coverage,
    }


def derive_schedule_markers(
    heating_plan: dict,
    cfg: UserConfig,
    model: ThermalModel,
    target_temp: float,
):
    timestamps = heating_plan.get("timestamps", pd.Series([], dtype="datetime64[ns]"))
    if not isinstance(timestamps, pd.Series):
        timestamps = pd.Series(timestamps)
    window_info = heating_plan.get("heating_windows", [])
    ramp_cap = heating_plan.get("max_ramp_c_per_hour")
    trigger_ts = None
    reach_ts = None
    if window_info and len(timestamps) > 0:
        first_window = window_info[0]
        preheat_idx = min(first_window["preheat_start"], len(timestamps) - 1)
        trigger_ts = timestamps.iloc[preheat_idx]
        start_idx = min(first_window["start_idx"], len(timestamps) - 1)
        start_ts = timestamps.iloc[start_idx]
        indoor_start = float(cfg.temp_current)
        try:
            indoor_start = float(heating_plan["indoor_series"].iloc[start_idx])
            eff_temp = float(
                heating_plan["temp_ext_series"].iloc[start_idx]
                - 0.2 * heating_plan["wind_series"].iloc[start_idx]
            )
            eta_hours = model.time_to_reach(
                indoor_start,
                target_temp,
                eff_temp,
                max_delta_per_hour=ramp_cap,
            )
        except Exception:
            eta_hours = None
        eta_hours = enforce_realistic_eta(indoor_start, target_temp, eta_hours)
        reach_ts = start_ts + pd.Timedelta(hours=max(0.0, eta_hours))
    return trigger_ts, reach_ts


def build_topline_summary(
    cfg: UserConfig,
    df: pd.DataFrame,
    heating_plan: dict,
    model: ThermalModel,
    stove_mode: str,
    target_temp: float,
    cost: float,
    pellet_cost: float,
    electric_cost: float,
    pellet_usage: dict | None = None,
):
    def _format_ts(ts: pd.Timestamp) -> str:
        return ts.strftime("%H:%M · %d/%m")

    def chip(text: str, tooltip: str) -> str:
        return (
            f"<span class='hero-chip' title='{html.escape(tooltip)}'>{html.escape(text)}</span>"
        )

    ext_temp = None
    if not df.empty and "temp_ext" in df:
        try:
            ext_temp = float(df["temp_ext"].iloc[0])
        except Exception:
            ext_temp = None

    timestamps = heating_plan.get("timestamps", pd.Series([], dtype="datetime64[ns]"))
    if not isinstance(timestamps, pd.Series):
        timestamps = pd.Series(timestamps)

    trigger_label = "dès que possible"
    reach_label = "--"
    trigger_ts, reach_ts = derive_schedule_markers(heating_plan, cfg, model, target_temp)
    if trigger_ts is not None:
        trigger_label = _format_ts(trigger_ts)
    if reach_ts is not None:
        reach_label = _format_ts(reach_ts)

    ext_str = f"{ext_temp:.1f}°C dehors" if ext_temp is not None else "température extérieure inconnue"
    horizon_hours = int(heating_plan.get("horizon", 24) or len(timestamps))
    planning_days = max(1, math.ceil(max(1, horizon_hours) / 24))
    if planning_days == 1:
        budget_scope_desc = "24 h"
    else:
        budget_scope_desc = f"{planning_days} jours ({planning_days * 24} h)"

    mode_chip = chip(stove_mode, "Réglage 'Mode d'utilisation du poêle'")
    target_chip = chip(f"{target_temp:.1f} °C", "Réglage 'Température intérieure cible'")
    trigger_chip = chip(trigger_label, "Détermination automatique sur la fenêtre de 24 h")
    reach_chip = chip(reach_label, "Projection issue de la courbe idéale ralentie")
    budget_chip = chip(f"{cost:.2f} €", f"Budget total estimé sur {budget_scope_desc}")
    pellet_cost_chip = chip(f"{pellet_cost:.2f} €", "Basé sur le prix du sac (15 kg) saisi")
    electric_chip = chip(f"{electric_cost:.2f} €", "Prix électricité (poêle + VMC)")
    volume_chip = chip(f"{cfg.volume_m3:.0f} m³", "Volume chauffé saisi")
    structure_chip = chip(cfg.structure, "Structure / paroi sélectionnée")

    pellet_usage = pellet_usage or {}
    pellet_total = pellet_usage.get("total_bags")
    pellet_usage_chip = (
        chip(f"{pellet_total:.2f} sac(s)", "Projection pellets sur l'horizon")
        if pellet_total is not None
        else ""
    )
    per_day_breakdown = ""
    per_day_values = pellet_usage.get("per_day") if pellet_total is not None else None
    if per_day_values:
        day_chunks = [f"Jour {idx + 1}: {value:.2f} sac" for idx, value in enumerate(per_day_values)]
        per_day_breakdown = " [" + " · ".join(day_chunks) + "]"

    summary_text = (
        f"{volume_chip} · {structure_chip} — {ext_str}. "
        f"{mode_chip} vise une consigne à {target_chip}, en respectant la bande 20-24 °C. "
        f"Déclencher vers {trigger_chip} pour atteindre {reach_chip}. "
        f"Budget estimé ({budget_scope_desc}) : {budget_chip} = pellets {pellet_cost_chip} + électricité {electric_chip}. "
    )
    if pellet_usage_chip:
        summary_text += f"Consommation pellets ({budget_scope_desc}) : {pellet_usage_chip}{per_day_breakdown}. "
    summary_text += "<span class='hero-note'>Hypothèse pellet : 1 sac (15 kg) ≈ 14 h de chauffe autour de 22°C.</span>"

    style = (
        "<style>"
        ".hero-summary {background:#ffffff;border:1px solid #dcdcdc;border-radius:12px;padding:22px;margin-bottom:24px;box-shadow:0 3px 10px rgba(0,0,0,0.06);}"
        ".hero-summary p {margin:0;font-size:1.08rem;line-height:1.6;color:#1f2933;}"
        ".hero-chip {display:inline-flex;align-items:center;background:#f5f7fa;border:1px solid #d7dde5;border-radius:999px;padding:2px 10px;font-size:0.94rem;margin:0 4px;white-space:nowrap;}"
        ".hero-chip:hover {background:#e3f2fd;border-color:#64b5f6;}"
        ".hero-note {display:block;margin-top:10px;font-size:0.85rem;color:#546e7a;}"
        "</style>"
    )
    return style + f"<div class='hero-summary'><p>{summary_text}</p></div>"


def render_green_dashboard(data: dict):
    temp_ext = float(data.get("temp_ext", 0.0))
    target_temp = float(data.get("temp_target", 21.0))
    temp_current = float(data.get("temp_current", target_temp))
    surface_label = data.get("surface_label", "N/A")
    isolation_label = data.get("isolation_label", "Isolation")
    trigger_label = data.get("heure_declenchement", "--")
    reach_label = data.get("heure_objectif", "--")
    pellets_48h = data.get("pellets_48h", 0.0)
    cost_pellets = data.get("cost_pellets", 0.0)
    cost_elec = data.get("cost_elec", 0.0)
    temp_bounds = data.get("temp_bounds", (16.0, 25.0))
    if "ui_temp_current" not in st.session_state:
        st.session_state["ui_temp_current"] = temp_current
    if "ui_temp_target" not in st.session_state:
        st.session_state["ui_temp_target"] = target_temp

    current_temp_state = float(st.session_state.get("ui_temp_current", temp_current))
    target_temp_state = float(st.session_state.get("ui_temp_target", target_temp))

    def _ensure_24h(label: str | None) -> str:
        if not label or label.strip() == "--":
            return "--"
        parsed = pd.to_datetime(label, errors="coerce")
        if isinstance(parsed, pd.Timestamp):
            return parsed.strftime("%H:%M")
        return label

    trigger_label = _ensure_24h(str(trigger_label))
    reach_label = _ensure_24h(str(reach_label))

    min_temp, max_temp = temp_bounds

    def update_session(key: str, value: float):
        new_val = float(value)
        prev = st.session_state.get(key)
        prev_val = float(prev) if prev is not None else None
        if prev_val is None or abs(prev_val - new_val) > 1e-6:
            st.session_state[key] = new_val
            _safe_rerun()

    st.markdown(
        """
        <style>
            .hero-condensed {background:#ffffff;border:1px solid #e2e8f0;border-radius:18px;padding:18px 22px;margin-bottom:26px;box-shadow:0 10px 30px rgba(15,23,42,0.08);} 
            .hero-condensed h3 {margin:0 0 8px 0;font-size:1.05rem;color:#0f172a;text-transform:uppercase;letter-spacing:0.1em;} 
            .hero-line {margin:4px 0;font-size:0.98rem;color:#0f172a;} 
            .hero-pill {margin-top:12px;padding:10px 14px;border-radius:12px;background:#f0fdf4;border:1px solid #bbf7d0;font-weight:600;color:#065f46;} 
            .hero-meta {margin-top:10px;font-size:0.9rem;color:#475569;} 
            .hero-note {margin-top:6px;font-size:0.78rem;color:#94a3b8;} 
            .hero-metrics {display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;font-size:0.9rem;color:#0f172a;} 
            .hero-metrics span {background:#f8fafc;border:1px solid #e2e8f0;border-radius:999px;padding:4px 10px;} 
        </style>
        """,
        unsafe_allow_html=True,
    )

    input_col, target_col, ext_col = st.columns([1.2, 1.2, 0.9])
    with input_col:
        current_value = st.number_input(
            "Température actuelle (°C)",
            min_value=float(min_temp - 5),
            max_value=float(max_temp + 5),
            value=float(current_temp_state),
            step=0.1,
            key="hero_temp_current_input",
        )
        update_session("ui_temp_current", float(current_value))
        current_temp_state = float(st.session_state.get("ui_temp_current", current_value))
    with target_col:
        target_value = st.number_input(
            "Température cible (°C)",
            min_value=float(min_temp),
            max_value=float(max_temp + 2),
            value=float(target_temp_state),
            step=0.1,
            key="hero_temp_target_input",
        )
        update_session("ui_temp_target", float(target_value))
        target_temp_state = float(st.session_state.get("ui_temp_target", target_value))
    with ext_col:
        st.metric("Extérieur", f"{temp_ext:.1f} °C")

    def estimate_standard_heating_time(temp_int: float, temp_target_val: float, temp_ext_val: float):
        if temp_target_val <= temp_int:
            return 0.0
        eta = STANDARD_HOME_MODEL.time_to_reach(
            temp_int,
            temp_target_val,
            temp_ext_val,
            max_delta_per_hour=STANDARD_HOME_SETTINGS["max_delta_per_hour"],
        )
        eta = enforce_realistic_eta(temp_int, temp_target_val, eta)
        return max(0.0, float(eta)) if eta is not None else None

    def format_duration(hours: float | None) -> str:
        if hours is None:
            return "indisponible"
        minutes = int(round(hours * 60))
        if minutes <= 10:
            return "< 10 min"
        hrs = minutes // 60
        mins = minutes % 60
        parts = []
        if hrs:
            parts.append(f"{hrs} h")
        if mins:
            parts.append(f"{mins} min")
        return " ".join(parts)

    eta_source_label = data.get("eta_source_label")
    eta_hours = data.get("eta_hours")
    if eta_hours is None:
        eta_hours = estimate_standard_heating_time(current_temp_state, target_temp_state, temp_ext)
        if eta_source_label is None:
            eta_source_label = "Scénario logement standard 100 m²"
    eta_str = format_duration(eta_hours)
    eta_source_label = eta_source_label or "Projection chauffage"
    eta_source_safe = html.escape(eta_source_label)

    st.markdown(
        f"""
        <div class='hero-condensed'>
            <h3>Situation thermique</h3>
            <div class='hero-line'>🏠 {surface_label} · 🧱 {html.escape(isolation_label)} · 🌡️ {temp_ext:.1f}°C dehors</div>
            <div class='hero-line'>Intérieur {current_temp_state:.1f}°C → Consigne {target_temp_state:.1f}°C · Allumer vers {trigger_label} pour viser {reach_label}</div>
            <div class='hero-pill'>{eta_source_safe} : <strong>{eta_str}</strong> de chauffe si vous démarrez maintenant</div>
            <div class='hero-metrics'>
                <span>Pellets 48h : {pellets_48h:.2f} sac</span>
                <span>Budget pellets : {cost_pellets:.2f} €</span>
                <span>Électricité : {cost_elec:.2f} €</span>
            </div>
            <div class='hero-meta'>Déclenchement {trigger_label} · Objectif atteint vers {reach_label}</div>
            <div class='hero-note'>Hypothèse standard : 100 m², {STANDARD_HOME_SETTINGS['power_kw']:.1f} kW @ {STANDARD_HOME_SETTINGS['efficiency']*100:.0f} %, τ = {STANDARD_HOME_SETTINGS['tau_hours']:.1f} h.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def build_vmc_legend_html(selected_vmc: str | None = None):
    rows = []
    for code, meta in VMC_METADATA.items():
        tooltip = html.escape(meta.get("tooltip", ""))
        label = html.escape(meta.get("label", code))
        dryness = html.escape(meta.get("dryness_hint", "Air sec"))
        annual = meta.get("annual_kwh", 0)
        row_class = "vmc-legend-row"
        if selected_vmc and code == selected_vmc:
            row_class += " vmc-selected"
        rows.append(
            "".join(
                [
                    f"<div class='{row_class}' data-vmc='{code}'>",
                    f"<abbr title=\"{tooltip}\">{code}</abbr>",
                    f"<span class='vmc-label'>{label}</span>",
                    f"<small>{annual} kWh/an · {dryness}</small>",
                    "</div>",
                ]
            )
        )
    rows_html = "\n".join(rows)
    style = (
        "<style>"
        ".vmc-legend {border:1px solid #d7d7d7;border-radius:6px;padding:8px;margin-top:6px;font-size:0.78rem;}"
        ".vmc-legend-row {display:flex;flex-direction:column;margin-bottom:6px;padding:4px;border-radius:4px;}"
        ".vmc-legend-row:last-child {margin-bottom:0;}"
        ".vmc-legend-row abbr {font-weight:600;cursor:help;text-decoration:none;border-bottom:1px dotted #999;}"
        ".vmc-label {font-size:0.75rem;color:#444;}"
        ".vmc-legend-row small {color:#666;}"
        ".vmc-legend-row.vmc-selected {background-color:#eef7ff;border:1px solid #90caf9;}"
        "</style>"
    )
    return style + f"<div class='vmc-legend'>{rows_html}</div>"


def apply_timeline_styling_day_night(fig: go.Figure, hours):
    if hours is None:
        values = []
    elif isinstance(hours, (pd.Series, pd.Index)):
        values = hours.tolist()
    else:
        try:
            values = list(hours)
        except TypeError:
            values = [hours]
    if not values:
        return fig

    try:
        ts = pd.to_datetime(pd.Series(values)).dropna().reset_index(drop=True)
    except Exception:
        return fig

    shapes = []
    if ts.empty:
        return fig

    # Highlight nighttime periods (before 6h and after 20h) for each day in range.
    night_mask = (ts.dt.hour <= 5) | (ts.dt.hour >= 20)
    if night_mask.any():
        start_idx = None
        for idx, is_night in enumerate(night_mask):
            if is_night and start_idx is None:
                start_idx = idx
            elif not is_night and start_idx is not None:
                shapes.append(
                    dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=ts.iloc[start_idx],
                        x1=ts.iloc[idx - 1],
                        y0=0,
                        y1=1,
                        fillcolor="rgba(20,20,60,0.08)",
                        line=dict(width=0),
                        layer="below",
                    )
                )
                start_idx = None
        if start_idx is not None:
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=ts.iloc[start_idx],
                    x1=ts.iloc[-1],
                    y0=0,
                    y1=1,
                    fillcolor="rgba(20,20,60,0.08)",
                    line=dict(width=0),
                    layer="below",
                )
            )

    current_label = _resolve_current_label(ts)
    if current_label is not None:
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


def add_day_separators(fig: go.Figure, timestamps, label_prefix="Instance", label_map: dict | None = None):
    if timestamps is None:
        return fig
    try:
        ts = pd.to_datetime(pd.Series(timestamps)).dropna().reset_index(drop=True)
    except Exception:
        return fig
    if ts.empty:
        return fig

    first_day = ts.iloc[0].normalize()
    last_day = ts.iloc[-1].normalize()
    current_day = first_day
    day_idx = 1
    while current_day <= last_day:
        next_day = current_day + pd.Timedelta(days=1)
        x1 = min(next_day, ts.iloc[-1])
        fig.add_vrect(
            xref="x",
            yref="paper",
            x0=current_day,
            x1=x1,
            fillcolor="rgba(33, 150, 243, {opacity})".format(opacity=0.04 if day_idx % 2 else 0.07),
            line_width=0,
            layer="below",
        )
        label_x = current_day + (x1 - current_day) / 2
        label_text = None
        if label_map:
            label_text = label_map.get(day_idx)
        if not label_text:
            label_text = f"{label_prefix} {day_idx}"

        fig.add_annotation(
            x=label_x,
            y=1.04,
            xref="x",
            yref="paper",
            text=label_text,
            showarrow=False,
            font=dict(size=11, color="#4a4a4a"),
        )
        if next_day <= ts.iloc[-1]:
            fig.add_vline(
                x=next_day,
                line=dict(color="#9e9e9e", width=1, dash="dot"),
                layer="below",
            )
        current_day = next_day
        day_idx += 1
    return fig


def summarize_instance_windows(timestamps, heating_windows, instance_hours=24):
    if timestamps is None or len(heating_windows) == 0:
        return []
    ts = pd.to_datetime(pd.Series(timestamps)).dropna().reset_index(drop=True)
    if ts.empty:
        return []
    base_day = ts.iloc[0].floor("D")
    total_hours = len(ts)
    instance_count = max(1, math.ceil(total_hours / instance_hours))
    summary = []
    for idx in range(instance_count):
        start = base_day + pd.Timedelta(hours=idx * instance_hours)
        end = start + pd.Timedelta(hours=instance_hours)
        count = 0
        for window in heating_windows:
            start_idx = min(window.get("start_idx", 0), len(ts) - 1)
            start_ts = ts.iloc[start_idx]
            if start <= start_ts < end:
                count += 1
        summary.append({"instance": idx + 1, "count": count, "start": start, "end": end})
    return summary


def enforce_realistic_eta(indoor_temp: float, target_temp: float, eta_hours: float | None) -> float:
    delta = max(0.0, target_temp - indoor_temp)
    baseline = max(3.0, delta * 1.5)
    if eta_hours is None:
        return baseline
    return max(eta_hours, baseline)


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
    horizon = min(PLANNING_HORIZON_HOURS, len(df))
    if horizon == 0:
        return {
            "horizon": 0,
            "heating_windows": [],
            "active_mask": [False] * PLANNING_HORIZON_HOURS,
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
    if mode_profile.get("strategy") == "pulse" and mode_profile.get("pulse_target_temp") is not None:
        target_temp = min(target_temp, float(mode_profile["pulse_target_temp"]))

    ramp_cap = float(mode_profile.get("max_delta_per_hour", 2.5))
    lower_guard = target_temp - 4.0
    floor_temp = mode_profile.get("floor_temp")
    restore_temp = mode_profile.get("restore_temp")
    if floor_temp is not None:
        lower_guard = max(lower_guard, float(floor_temp))
    if restore_temp is not None:
        lower_guard = max(lower_guard, float(restore_temp))
    comfort_upper = target_temp + 4.0
    restore_for_calc = float(restore_temp) if restore_temp is not None else lower_guard

    def _minutes_to_steps(minutes: float, ensure_full_hour: bool = False) -> int:
        if minutes <= 0:
            return 0
        if minutes < 60:
            return 1 if ensure_full_hour else 0
        return int(math.ceil(minutes / 60.0))

    base_preheat_minutes = 60.0
    warm_preheat_minutes = 15.0
    base_preheat_steps = max(1, _minutes_to_steps(base_preheat_minutes, ensure_full_hour=True))
    warm_preheat_steps = _minutes_to_steps(warm_preheat_minutes, ensure_full_hour=False)
    preheat_hours = base_preheat_minutes / 60.0

    cooldown_minutes = mode_profile.get("cooldown_minutes", 30)
    cooldown_steps = max(0, _minutes_to_steps(float(cooldown_minutes), ensure_full_hour=False))

    daily_hours_budget = float(hours_on) if hours_on else mode_profile["default_total_hours"]
    if mode_profile["strategy"] == "pulse":
        daily_hours_budget = mode_profile["default_total_hours"]
    daily_hours_budget = max(1.0, min(24.0, daily_hours_budget))
    min_sessions = max(1, int(mode_profile.get("min_sessions", 1)))
    max_sessions_cfg = max(min_sessions, int(mode_profile.get("max_sessions", min_sessions)))
    max_session_hours = max(1, int(mode_profile.get("max_session_hours", 1)))
    sessions_hint = max(1, int(math.ceil(daily_hours_budget / max_session_hours)))
    sessions_per_instance = min(max_sessions_cfg, max(min_sessions, sessions_hint))
    instances = max(1, math.ceil(horizon / 24))
    target_sessions = sessions_per_instance * instances
    per_session = int(math.floor(daily_hours_budget / sessions_per_instance)) if sessions_per_instance else int(daily_hours_budget)
    if per_session <= 0:
        per_session = 1
    session_hours = max(1, min(max_session_hours, per_session))
    anchor_hours = list(mode_profile.get("anchor_hours") or [])

    ext_eff_series = (temp_ext_series - 0.2 * wind_series).reset_index(drop=True)

    if mode_profile.get("strategy") == "pulse":
        try:
            eff_ext = float(ext_eff_series.mean()) if not ext_eff_series.empty else float(temp_ext_series.iloc[0])
        except Exception:
            eff_ext = float(temp_ext_series.iloc[0]) if len(temp_ext_series) else 5.0
        eta_hours = model.time_to_reach(
            restore_for_calc,
            target_temp,
            eff_ext,
            max_delta_per_hour=ramp_cap,
        )
        eta_hours = enforce_realistic_eta(restore_for_calc, target_temp, eta_hours)
        required_session_hours = max(1, int(math.ceil(eta_hours)))
        session_hours = max(
            session_hours,
            min(max_session_hours, required_session_hours),
        )

    def simulate_mask(mask, clamp=False):
        series = model.simulate(
            float(cfg.temp_current),
            temp_ext_series.values,
            wind_series.values,
            hours_on=0,
            humidity_series=humidity_series.values if humidity_series is not None else None,
            target_temp=target_temp if clamp else None,
            custom_mask=mask,
            max_delta_per_hour=ramp_cap,
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
    temp_series = simulate_mask(active_mask, clamp=False)
    last_run_end_idx = None

    def build_day_anchors(day_idx: int):
        anchors = [float(anchor) + day_idx * 24 for anchor in anchor_hours]
        if len(anchors) >= sessions_per_instance:
            return sorted(anchors[:sessions_per_instance])
        missing = max(0, sessions_per_instance - len(anchors))
        day_start = day_idx * 24
        reference_horizon = horizon if horizon > 0 else 24
        day_end = min((day_idx + 1) * 24, reference_horizon) - 1
        if day_end <= day_start:
            day_end = day_start + 23
        fallback = []
        if missing > 0:
            early = day_start + 4
            late = max(early + 1, day_end - 2)
            fallback = np.linspace(early, late, missing + 2)[1:-1].tolist()
        return sorted((anchors + fallback)[:sessions_per_instance])

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
        ext_window = ext_eff_series.loc[window_start:window_end]
        if not ext_window.empty:
            return int(ext_window.idxmin())
        future_hits = temp_series.loc[target_idx:][temp_series.loc[target_idx:] < lower_guard]
        if not future_hits.empty:
            return int(future_hits.index[0])
        past_window = temp_series.loc[:target_idx]
        past_hits = past_window[past_window < lower_guard]
        if not past_hits.empty:
            return int(past_hits.index[-1])
        return target_idx

    def schedule_window(run_start_idx: int):
        nonlocal temp_series, last_run_end_idx
        run_start = max(0, min(horizon - 1, run_start_idx))
        if last_run_end_idx is not None:
            min_start_idx = last_run_end_idx + 1
            if cooldown_steps > 0:
                min_start_idx = max(min_start_idx, last_run_end_idx + cooldown_steps + 1)
            run_start = max(run_start, min_start_idx)
        if run_start >= horizon:
            return False
        warm_start = last_run_end_idx is not None and (run_start - last_run_end_idx) <= 1
        preheat_minutes = warm_preheat_minutes if warm_start else base_preheat_minutes
        preheat_steps = warm_preheat_steps if warm_start else base_preheat_steps
        preheat_start = max(0, run_start - preheat_steps)
        run_end = min(horizon - 1, run_start + session_hours - 1)
        for idx in range(run_start, run_end + 1):
            active_mask[idx] = True
        heating_windows.append(
            {
                "preheat_start": preheat_start,
                "start_idx": run_start,
                "end_idx": run_end,
                "cooldown_minutes": cooldown_minutes,
                "preheat_minutes": preheat_minutes,
                "warm_start": warm_start,
            }
        )
        last_run_end_idx = run_end
        temp_series = simulate_mask(active_mask, clamp=False)
        return True

    selected_anchor_hours = []
    for day_idx in range(instances):
        selected_anchor_hours.extend(build_day_anchors(day_idx))
    if not selected_anchor_hours:
        selected_anchor_hours = np.linspace(0, max(horizon - 1, 1), target_sessions).tolist()

    if mode_profile["strategy"] == "pulse":
        for anchor_hour in selected_anchor_hours:
            anchor_idx = locate_run_start(anchor_hour)
            if temp_series.empty:
                start_idx = anchor_idx
            else:
                min_idx = int(temp_series.idxmin())
                anchor_temp = temp_series.iloc[anchor_idx] if anchor_idx < len(temp_series) else None
                min_temp = temp_series.iloc[min_idx]
                start_idx = anchor_idx if (anchor_temp is not None and anchor_temp <= min_temp + 0.3) else min_idx
            if schedule_window(start_idx):
                planned_sessions += 1
            if planned_sessions >= target_sessions:
                break
    else:
        for anchor_hour in selected_anchor_hours:
            run_start_idx = locate_run_start(anchor_hour)
            if schedule_window(run_start_idx):
                planned_sessions += 1
            if planned_sessions >= target_sessions:
                break

    indoor_series = simulate_mask(active_mask, clamp=True).clip(lower=-50, upper=50)
    active_hours = sum(1 for flag in active_mask[:horizon] if flag)

    return {
        "horizon": horizon,
        "heating_windows": heating_windows,
        "active_mask": active_mask,
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
        "preheat_base_minutes": base_preheat_minutes,
        "preheat_warm_minutes": warm_preheat_minutes,
        "cooldown_minutes": cooldown_minutes,
        "mode_label": stove_mode,
        "planned_sessions": planned_sessions,
        "session_hours": session_hours,
        "max_ramp_c_per_hour": ramp_cap,
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

    def vmc_format(key):
        meta = VMC_METADATA.get(key, {})
        label = meta.get("label")
        return f"{key} — {label}" if label else key

    vmc = st.sidebar.selectbox(
        "VMC",
        vmc_options,
        index=vmc_index,
        format_func=vmc_format,
        help="La ventilation extrait l'air humide pour permettre au poêle de chauffer un air plus sec.",
    )
    vmc_meta = VMC_METADATA.get(vmc, {})
    tooltip = vmc_meta.get("tooltip", "")
    if tooltip:
        st.sidebar.caption(tooltip)
    st.sidebar.markdown(build_vmc_legend_html(selected_vmc=vmc), unsafe_allow_html=True)

    power = st.sidebar.number_input("Puissance poêle (kW)", min_value=1.0, max_value=20.0, value=8.0)
    efficiency = st.sidebar.slider("Rendement", min_value=0.5, max_value=1.0, value=0.85)
    if "ui_temp_current" not in st.session_state:
        st.session_state["ui_temp_current"] = 19.0
    temp_current = st.sidebar.number_input(
        "Température intérieure actuelle",
        value=float(st.session_state["ui_temp_current"]),
    )
    if abs(temp_current - st.session_state.get("ui_temp_current", temp_current)) > 1e-3:
        st.session_state["ui_temp_current"] = float(temp_current)
    if "ui_temp_target" not in st.session_state:
        st.session_state["ui_temp_target"] = 22.0
    temp_target = float(st.session_state["ui_temp_target"])
    st.sidebar.caption(f"Consigne ajustée via le slider : {temp_target:.1f}°C")

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
        help="Confort = 3 à 4 cycles serrés, Eco = 2 cycles sobres, Vacances = impulsion quotidienne automatique.",
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

    vmc_annual_kwh = VMC_ANNUAL_KWH.get(cfg.vmc, 0.0)
    vmc_daily_kwh = vmc_annual_kwh / 365.0
    vmc_daily_cost = vmc_daily_kwh * electric_price
    stove_electric_cost = kpi.stove_electric_cost(hours_on)

    cost = base_cost * weather_factor + vmc_daily_cost
    loss = base_loss * weather_factor
    pellet_cost = kpi.pellet_cost(pellet_df)
    electric_cost = stove_electric_cost + vmc_daily_cost
    meteo_stats = {
        "wind_mean": wind_mean,
        "humidity_mean": humidity_mean,
        "temp_mean": temp_mean,
        "deg_day": deg,
        "weather_factor": weather_factor,
        "vmc_daily_kwh": vmc_daily_kwh,
        "vmc_daily_cost": vmc_daily_cost,
        "vmc_label": cfg.vmc,
        "stove_electric_cost": stove_electric_cost,
    }
    return cost, deg, loss, pellet_cost, electric_cost, meteo_stats


def render_weather_tab(df, pellet_df, hours_on, heating_plan):
    st.subheader("Sources météo et projections associées")
    horizon = min(PLANNING_HORIZON_HOURS, len(df))
    if horizon == 0:
        st.warning("Impossible d'afficher les données météo.")
        return

    hours_axis = df["time"].head(horizon)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    plan_windows = heating_plan.get("heating_windows", []) if heating_plan else []
    plan_timestamps = heating_plan.get("timestamps") if heating_plan else None
    if plan_timestamps is not None and not isinstance(plan_timestamps, pd.Series):
        plan_timestamps = pd.Series(plan_timestamps)
    plan_cooldown = heating_plan.get("cooldown_minutes", 0) if heating_plan else 0
    instance_summary = summarize_instance_windows(plan_timestamps, plan_windows)
    if instance_summary:
        summary_text = " · ".join(
            [f"Instance {item['instance']}: {item['count']} allumage(s)" for item in instance_summary]
        )
        st.caption(f"Répartition planifiée — {summary_text}")

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
        title=f"Fresque météo ({horizon}h)",
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
    if plan_windows and plan_timestamps is not None and len(plan_timestamps) >= 1:
        for window in plan_windows:
            start_idx = min(window["start_idx"], len(plan_timestamps) - 1)
            end_idx = min(window["end_idx"], len(plan_timestamps) - 1)
            preheat_idx = min(window["preheat_start"], len(plan_timestamps) - 1)
            start_time = plan_timestamps.iloc[start_idx]
            end_time = plan_timestamps.iloc[end_idx]
            preheat_time = plan_timestamps.iloc[preheat_idx]
            cooldown_end = end_time + pd.Timedelta(minutes=window.get("cooldown_minutes", plan_cooldown))
            fig.add_vrect(
                x0=preheat_time,
                x1=start_time,
                fillcolor="rgba(255,183,77,0.15)",
                line_width=0,
                layer="below",
            )
            fig.add_vrect(
                x0=start_time,
                x1=end_time,
                fillcolor="rgba(255,87,34,0.18)",
                line_width=0,
                layer="below",
            )
            fig.add_vrect(
                x0=end_time,
                x1=cooldown_end,
                fillcolor="rgba(76,175,80,0.12)",
                line_width=0,
                layer="below",
            )
    fig = apply_timeline_styling_day_night(fig, hours_axis)
    fig = add_day_separators(fig, hours_axis)
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
    base_preheat_minutes = heating_plan.get("preheat_base_minutes", preheat_hours * 60)
    warm_preheat_minutes = heating_plan.get("preheat_warm_minutes", 15.0)
    cooldown_minutes = heating_plan["cooldown_minutes"]
    ramp_cap = heating_plan.get("max_ramp_c_per_hour")
    instance_summary = summarize_instance_windows(timestamps, heating_windows)
    if instance_summary:
        day_chunks = []
        for item in instance_summary:
            label = "Jour" if item["instance"] == 1 else f"Jour+{item['instance'] - 1}"
            day_chunks.append(f"{label} : {item['count']} allumage(s)")
        summary_text = " · ".join(day_chunks)
        st.caption(f"Distribution sur 24h — {summary_text}")

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
    recharge_labels = []
    if "recharge" in pellet_df:
        recharge_idx = [int(i) for i in pellet_df.index[pellet_df["recharge"]]]
        recharge_hours = [hours_axis[i] for i in recharge_idx if i < len(hours_axis)]
        recharge_labels = [int(pellet_df.iloc[i]["recharge_points"]) for i in recharge_idx if i < len(pellet_df)]
    if recharge_hours:
        marker_text = [f"Recharge sac #{val}" for val in recharge_labels]
        fig.add_trace(
            go.Scatter(
                x=recharge_hours,
                y=[timeline_df["Température intérieure projetée"].min() - 0.5] * len(recharge_hours),
                mode="markers+text",
                text=marker_text,
                textposition="top center",
                marker_symbol="triangle-up",
                marker_color="#e67e22",
                name="Points de recharge",
                hovertemplate="%{text}<extra></extra>",
            ),
            secondary_y=False,
        )
        recharge_caption = ", ".join([
            f"{txt} vers {ts:%d/%m %H:%M}" for txt, ts in zip(marker_text, recharge_hours)
        ])
        if recharge_caption:
            st.caption(f"Recharges granulés estimées : {recharge_caption}.")

    fig.update_layout(
        xaxis=dict(domain=[0, 0.95]),
        legend_title="Courbes",
        title=f"Fresque temporelle unifiée ({horizon}h)",
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
    instance_count = max(1, math.ceil(horizon / 24))
    label_map = {}
    for idx in range(1, instance_count + 1):
        label_map[idx] = "Jour" if idx == 1 else f"Jour+{idx - 1}"
    fig = add_day_separators(fig, timestamps, label_prefix="Jour", label_map=label_map)

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
        max_delta_per_hour=ramp_cap,
    )
    if heating_path:
        curve_y = [float(cfg.temp_current)] + heating_path
        curve_y = [min(target_temp, y) for y in curve_y]
        steps = max(1, len(curve_y) - 1)
        realistic_eta = enforce_realistic_eta(float(cfg.temp_current), target_temp, steps)
        step_hours = realistic_eta / steps
        curve_x = [now_marker + pd.Timedelta(hours=i * step_hours) for i in range(len(curve_y))]
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
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=1.02,
            y=1.02,
            text="Courbe idéale ⓘ",
            showarrow=False,
            font=dict(size=11, color="#7f8c8d"),
            align="left",
            hovertext="Si le poêle démarre maintenant, la courbe idéale estime le temps pour atteindre la consigne.",
            hoverlabel=dict(bgcolor="#fef5e7", font=dict(color="#4a4a4a")),
            captureevents=True,
        )
        reach_time = curve_x[-1]
        st.caption(f"Si le poêle démarre maintenant, la consigne serait atteinte vers {reach_time:%H:%M} (modèle ralenti).")

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
                annotation_text="Pré-chauffe",
                annotation_position="top left",
            )

        fig.add_vrect(
            x0=start_time,
            x1=end_time,
            fillcolor="rgba(255,87,34,0.28)",
            line_width=0,
            layer="below",
            annotation_text="Chauffe nominale",
            annotation_position="top right",
        )

        fig.add_vrect(
            x0=end_time,
            x1=cooldown_end,
            fillcolor="rgba(76,175,80,0.20)",
            line_width=0,
            layer="below",
            annotation_text="Refroidissement",
            annotation_position="bottom right",
        )

    st.plotly_chart(fig, use_container_width=True)

    if heating_windows:
        st.caption(
            f"Bande de confort anticipée : {comfort_lower:.1f}°C – {comfort_upper:.1f}°C. Pré-chauffe nominale {preheat_hours:.1f} h (≈{int(base_preheat_minutes)} min), réduite à {int(warm_preheat_minutes)} min si relance < 1 h."
        )
        table_rows = []
        for idx, window in enumerate(heating_windows, start=1):
            preheat_time = timestamps.iloc[window["preheat_start"]]
            start_time = timestamps.iloc[window["start_idx"]]
            end_time = timestamps.iloc[min(window["end_idx"], len(timestamps) - 1)]
            cooldown_end = end_time + pd.Timedelta(minutes=window.get("cooldown_minutes", cooldown_minutes))
            preheat_minutes = float(window.get("preheat_minutes", base_preheat_minutes))
            preheat_duration = preheat_minutes / 60.0
            nominal_duration = max(1, window["end_idx"] - window["start_idx"] + 1)
            eff_temp = ext_eff_series.iloc[min(window["start_idx"], len(ext_eff_series) - 1)]
            eta = model.time_to_reach(
                indoor_series.iloc[window["start_idx"]],
                target_temp,
                eff_temp,
                max_delta_per_hour=ramp_cap,
            )
            time_to_target = enforce_realistic_eta(indoor_series.iloc[window["start_idx"]], target_temp, eta)
            table_rows.append(
                {
                    "Fenêtre": idx,
                    "Pré-chauffe": preheat_time.strftime("%H:%M"),
                    "Allumage": start_time.strftime("%H:%M"),
                    "Arrêt estimé": end_time.strftime("%H:%M"),
                    "Fin refroidissement": cooldown_end.strftime("%H:%M"),
                    "Durée préchauffe (h)": round(preheat_duration, 2),
                    "Pré-chauffe (min)": int(preheat_minutes),
                    "Durée nominale (h)": nominal_duration,
                    "Temps pour atteindre la consigne (h)": round(time_to_target, 2) if time_to_target else "-",
                }
            )
        st.subheader("Fenêtres de chauffe prévues")
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)
    else:
        st.info(f"Aucune plage de chauffe anticipée n'est requise sur les prochaines {horizon} h.")



def render_kpi_tab(cost, deg, loss, pellet_cost, electric_cost, meteo_stats, pellet_usage):
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

    pellet_usage = pellet_usage or {"total_bags": 0.0, "per_day": []}
    st.metric(
        "Consommation pellets (horizon)",
        f"{pellet_usage.get('total_bags', 0.0):.2f} sac(s)",
        f"{pellet_usage.get('total_kg', 0.0):.1f} kg",
    )
    if pellet_usage.get("per_day"):
        per_day_txt = " · ".join(
            [f"Jour {idx + 1}: {bags:.2f} sac" for idx, bags in enumerate(pellet_usage["per_day"])]
        )
        st.caption(f"Détail sur {pellet_usage.get('coverage_hours', 0)} h — {per_day_txt}.")

    vmc_cost = meteo_stats.get("vmc_daily_cost", 0.0)
    vmc_kwh = meteo_stats.get("vmc_daily_kwh", 0.0)
    vmc_label = meteo_stats.get("vmc_label", "VMC")
    stove_elec_cost = meteo_stats.get("stove_electric_cost", electric_cost - vmc_cost)
    st.markdown(
        f"Coût pellet: **{pellet_cost:.2f} €** — Élec poêle: **{stove_elec_cost:.2f} €** — VMC {vmc_label}: **{vmc_cost:.2f} €** ({vmc_kwh:.2f} kWh/j)",
        help="Un air intérieur plus sec (grâce à la VMC) se réchauffe plus vite : on additionne donc son coût électrique au budget.",
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

    render_circular_performance_chart(score)

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
    st.dataframe(comparison_df, hide_index=True, use_container_width=True)
    st.caption(f"Références régionales calculées autour de {region_label}.")


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



CLIMATE_VARIABLES = {
    "Température": {
        "label": "Température extérieure",
        "column": "temp",
        "unit": "°C",
        "colorscale": [
            [0.0, "#0ea5e9"],
            [1.0, "#dc2626"],
        ],
        "value_range": None,
        "pill_color": "#dc2626",
        "heat_opacity": 0.55,
        "pill_opacity": 0.7,
    },
    "Humidité": {
        "label": "Humidité relative",
        "column": "humidity",
        "unit": "%",
        "colorscale": [
            [0.0, "#ecfccb"],
            [1.0, "#14532d"],
        ],
        "value_range": None,
        "pill_color": "#15803d",
        "heat_opacity": 0.45,
        "pill_opacity": 0.65,
    },
    "Vent": {
        "label": "Vent moyen",
        "column": "wind",
        "unit": "km/h",
        "colorscale": [
            [0.0, "#e0f2fe"],
            [1.0, "#312e81"],
        ],
        "value_range": None,
        "pill_color": "#312e81",
        "heat_opacity": 0.5,
        "pill_opacity": 0.7,
    },
    "Rayonnement": {
        "label": "Rayonnement solaire",
        "column": "radiation",
        "unit": "W/m²",
        "colorscale": [
            [0.0, "#fff7ed"],
            [1.0, "#ea580c"],
        ],
        "value_range": None,
        "pill_color": "#ea580c",
        "heat_opacity": 0.5,
        "pill_opacity": 0.7,
    },
}

MAP_SCALE_CONFIG = {
    "Ville": {"group": "city", "zoom": 6.8, "density_radius": 18},
    "Commune": {"group": "city", "zoom": 6.2, "density_radius": 22},
    "Département": {"group": "department", "zoom": 5.0, "density_radius": 35},
    "Région": {"group": "region", "zoom": 4.3, "density_radius": 45},
}

MAP_LAYER_CHOICES = ["Heatmap", "Fournisseurs", "Heatmap + Fournisseurs"]

FRANCE_BOUNDARY_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"name": "France"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-5.2, 51.5],
                    [-5.2, 42.0],
                    [9.8, 42.0],
                    [9.8, 51.5],
                    [-5.2, 51.5],
                ]],
            },
        }
    ],
}


def load_fournisseurs_data(csv_path: str = "data/fournisseurs.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        df = pd.DataFrame(
            [
                {
                    "nom": "Pellet Vert",
                    "region": "Auvergne-Rhône-Alpes",
                    "latitude": 45.76,
                    "longitude": 4.83,
                    "logo_url": "",
                    "site_web": "https://pelletvert.example.com",
                }
            ]
        )
    df = df.dropna(subset=["latitude", "longitude"])
    return df


def build_climate_snapshot(communes: pd.DataFrame, frames: int = 1) -> tuple[pd.DataFrame, pd.Series | list | None]:
    fm = FranceMeteo()
    weather = fm.fetch(communes, frames=max(1, frames))
    data = weather["data"].copy()
    timeline = weather.get("timeline")

    base_columns = ["city", "lat", "lon", "region", "department", "pop"]
    metrics = [
        ("temp", "temp"),
        ("wind", "wind"),
        ("humidity", "humidity"),
        ("radiation", "radiation"),
        ("wind_dir", "wind_dir"),
    ]

    records = []
    frame_count = max(1, frames)
    for idx in range(frame_count):
        frame_df = data[base_columns].copy()
        for alias, prefix in metrics:
            source_col = f"{prefix}_{idx}"
            frame_df[alias] = data[source_col] if source_col in data.columns else np.nan
        frame_df["frame"] = idx
        if timeline is not None and idx < len(timeline):
            frame_df["timestamp"] = timeline[idx]
        records.append(frame_df)

    if records:
        snapshot = pd.concat(records, ignore_index=True)
    else:
        snapshot = pd.DataFrame(columns=base_columns + [m[0] for m in metrics] + ["frame", "timestamp"])

    snapshot.dropna(subset=["lat", "lon"], inplace=True)
    if "pop" in snapshot.columns:
        snapshot["pop"] = snapshot["pop"].fillna(0)
    else:
        snapshot["pop"] = pd.Series(0, index=snapshot.index, dtype=float)
    return snapshot, timeline


def aggregate_climate(df: pd.DataFrame, scale: str) -> pd.DataFrame:
    config = MAP_SCALE_CONFIG.get(scale, MAP_SCALE_CONFIG["Région"])
    group_col = config["group"]
    if scale in ("Ville", "Commune"):
        view = df.copy()
        view["label"] = view["city"].astype(str)
        return view
    aggregations = {
        "temp": "mean",
        "wind": "mean",
        "humidity": "mean",
        "radiation": "mean",
        "lat": "mean",
        "lon": "mean",
        "region": "first",
        "department": "first",
        "pop": "sum",
    }
    include_wind_dir = "wind_dir" in df.columns
    working_df = df
    if include_wind_dir:
        working_df = df.copy()
        working_df["_wind_dir_rad"] = np.deg2rad(working_df["wind_dir"].astype(float))
        working_df["_wind_dir_sin"] = np.sin(working_df["_wind_dir_rad"])
        working_df["_wind_dir_cos"] = np.cos(working_df["_wind_dir_rad"])
        aggregations["wind_dir"] = "mean"
        aggregations["_wind_dir_sin"] = "mean"
        aggregations["_wind_dir_cos"] = "mean"
    aggregations.pop(group_col, None)
    agg = (
        working_df.groupby(group_col, as_index=False)
        .agg(aggregations)
    )
    if include_wind_dir:
        agg["wind_dir"] = np.degrees(np.arctan2(agg["_wind_dir_sin"], agg["_wind_dir_cos"])) % 360
        agg.drop(columns=["_wind_dir_sin", "_wind_dir_cos"], inplace=True, errors="ignore")
    agg["label"] = agg[group_col].astype(str)
    return agg


def haversine_distance(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def resolve_admin_level(lat: float, lon: float, base_df: pd.DataFrame) -> pd.Series | None:
    if base_df.empty:
        return None
    deltas = (base_df["lat"] - lat) ** 2 + (base_df["lon"] - lon) ** 2
    idx = int(deltas.idxmin())
    return base_df.iloc[idx]


def nearest_fournisseurs(lat: float, lon: float, providers: pd.DataFrame, user_position=None, top_n: int = 3) -> pd.DataFrame:
    if providers.empty:
        return pd.DataFrame()
    df = providers.copy()
    df["distance_click_km"] = df.apply(lambda row: haversine_distance(lat, lon, row["latitude"], row["longitude"]), axis=1)
    if user_position:
        df["distance_user_km"] = df.apply(
            lambda row: haversine_distance(user_position[0], user_position[1], row["latitude"], row["longitude"]), axis=1
        )
    else:
        df["distance_user_km"] = np.nan
    return df.nsmallest(top_n, "distance_click_km")


def render_base_map(scale: str, center: tuple[float, float]) -> go.Figure:
    zoom = MAP_SCALE_CONFIG.get(scale, MAP_SCALE_CONFIG["Région"]).get("zoom", 4.3)
    lat, lon = center
    if any(pd.isna([lat, lon])):
        lat, lon = 46.2276, 2.2137
    fig = go.Figure()
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            zoom=zoom,
            center={"lat": lat, "lon": lon},
            layers=[
                {
                    "sourcetype": "geojson",
                    "source": FRANCE_BOUNDARY_GEOJSON,
                    "type": "line",
                    "color": "#A8FF60",
                    "line": {"width": 2},
                }
            ],
        ),
        height=640,
        width=880,
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#f8fafc",
        margin=dict(l=0, r=0, t=10, b=10),
        font=dict(color="#0f172a"),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", font=dict(color="#0f172a")),
    )
    return fig


def render_heatmap_layer(
    fig: go.Figure,
    df: pd.DataFrame,
    variable: str,
    scale: str,
    *,
    name: str | None = None,
    visible: bool = True,
    show_colorbar: bool = True,
):
    if df.empty:
        return fig

    config = CLIMATE_VARIABLES[variable]
    column = config["column"]
    if column not in df.columns:
        return fig
    values = df[column]
    if values.empty:
        return fig

    radius = MAP_SCALE_CONFIG.get(scale, MAP_SCALE_CONFIG["Région"]).get("density_radius", 25)
    values_clean = values.dropna()
    value_range = config.get("value_range")
    zmin, zmax = (None, None)
    if isinstance(value_range, tuple):
        zmin, zmax = value_range
    elif not values_clean.empty:
        lower = float(values_clean.quantile(0.05)) if len(values_clean) > 1 else float(values_clean.min())
        upper = float(values_clean.quantile(0.95)) if len(values_clean) > 1 else float(values_clean.max())
        if math.isclose(lower, upper):
            lower = float(values_clean.min())
            upper = float(values_clean.max())
        zmin, zmax = lower, upper
    intensity_kwargs: dict[str, float] = {}
    if zmin is not None:
        intensity_kwargs["zmin"] = zmin
    if zmax is not None:
        intensity_kwargs["zmax"] = zmax

    label = config.get("label", variable)

    heat_opacity = float(config.get("heat_opacity", 0.6))

    fig.add_trace(
        go.Densitymapbox(
            lat=df["lat"],
            lon=df["lon"],
            z=values,
            radius=radius,
            colorscale=config["colorscale"],
            opacity=heat_opacity,
            colorbar=(
                dict(
                    title=dict(text=f"{label} ({config['unit']})", font=dict(color="#0f172a")),
                    tickcolor="#0f172a",
                )
                if show_colorbar
                else None
            ),
            showscale=show_colorbar,
            hovertemplate="<b>%{text}</b><br>" + f"{label}: %{{z:.1f}} {config['unit']}<extra></extra>",
            text=df.get("label", df.get("city", "")),
            name=name or f"{label} — {scale}",
            visible=visible,
            **intensity_kwargs,
        )
    )
    return fig


def render_metric_pills(
    fig: go.Figure,
    df: pd.DataFrame,
    variable: str,
    *,
    visible: bool = True,
    granularity: str = "Région",
):
    if df.empty:
        return fig
    config = CLIMATE_VARIABLES[variable]
    column = config["column"]
    if column not in df.columns:
        return fig
    df = df.dropna(subset=[column]).copy()
    if df.empty:
        return fig
    label = config.get("label", variable)
    text = [f"{row['label']}<br>{row[column]:.1f} {config['unit']}" for _, row in df.iterrows()]
    color = config.get("pill_color", "#2563eb")
    pill_opacity = float(config.get("pill_opacity", 0.75))
    fig.add_trace(
        go.Scattermapbox(
            lat=df["lat"],
            lon=df["lon"],
            mode="markers+text",
            text=text,
            textposition="top center",
            marker=dict(size=16, color=color, opacity=pill_opacity, symbol="circle"),
            name=f"{label} — {granularity}",
            hovertemplate="%{text}<extra></extra>",
            visible=visible,
        )
    )
    return fig

def render_fournisseur_layer(fig: go.Figure, providers: pd.DataFrame, *, visible: bool = False):
    if providers.empty:
        return fig

    site_series = providers["site_web"] if "site_web" in providers.columns else pd.Series([""] * len(providers))
    custom = np.stack(
        [
            providers["nom"].astype(str).to_numpy(),
            providers["region"].astype(str).to_numpy(),
            site_series.astype(str).to_numpy(),
        ],
        axis=-1,
    )

    fig.add_trace(
        go.Scattermapbox(
            lat=providers["latitude"],
            lon=providers["longitude"],
            mode="markers",
            marker=dict(size=18, color="#f97316", opacity=0.9, symbol="triangle"),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Région : %{customdata[1]}<br>"
                "<a href='%{customdata[2]}' target='_blank'>Site web</a><extra></extra>"
            ),
            name="Fournisseurs (calque)",
            visible=visible,
        )
    )
    return fig


def _bearing_to_arrow(degrees: float | None) -> str:
    if degrees is None or pd.isna(degrees):
        return "·"
    normalized = float(degrees) % 360
    sectors = [
        (22.5, "↑"),
        (67.5, "↗"),
        (112.5, "→"),
        (157.5, "↘"),
        (202.5, "↓"),
        (247.5, "↙"),
        (292.5, "←"),
        (337.5, "↖"),
    ]
    for limit, symbol in sectors:
        if normalized < limit:
            return symbol
    return "↑"


def render_wind_vector_layer(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    visible: bool = False,
    granularity: str = "Région",
    max_points: int = 150,
):
    if not visible or df is None or df.empty:
        return fig
    required = {"lat", "lon", "wind", "wind_dir"}
    if not required.issubset(df.columns):
        return fig
    subset = df.dropna(subset=list(required)).copy()
    if subset.empty:
        return fig
    subset = subset.sort_values("wind", ascending=False).head(max_points)
    arrows = [_bearing_to_arrow(val) for val in subset["wind_dir"]]
    texts = [f"{arrow} {speed:.0f} km/h" for arrow, speed in zip(arrows, subset["wind"], strict=False)]
    custom = np.stack(
        [
            subset["wind"].astype(float).to_numpy(),
            subset["wind_dir"].astype(float).to_numpy(),
        ],
        axis=-1,
    )
    fig.add_trace(
        go.Scattermapbox(
            lat=subset["lat"],
            lon=subset["lon"],
            mode="markers+text",
            text=texts,
            textposition="top center",
            marker=dict(
                size=np.clip(subset["wind"].astype(float), 8, 28),
                color="#0ea5e9",
                opacity=0.85,
                symbol="circle",
            ),
            name=f"Vecteur vent — {granularity}",
            customdata=custom,
            hovertemplate="Vent moyen: %{customdata[0]:.1f} km/h<br>Direction: %{customdata[1]:.0f}°<extra></extra>",
            visible=visible,
        )
    )
    return fig




def render_france_map(
    df_climat: pd.DataFrame,
    df_fournisseurs: pd.DataFrame,
    timeline,
    user_position=None,
):
    st.subheader("Radar météo national")
    if df_climat.empty:
        st.info("Impossible de charger les données climatiques pour la carte.")
        return

    st.markdown(
        """
        <style>
            .map-card {background:#ffffff;border:1px solid #e2e8f0;border-radius:18px;padding:18px 22px;margin-bottom:26px;box-shadow:0 16px 40px rgba(15,23,42,0.08);} 
            .map-card .header {display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px;}
            .map-card .header span {font-size:0.95rem;color:#475569;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    frames = sorted(df_climat["frame"].unique())
    min_frame, max_frame = int(min(frames)), int(max(frames))

    with st.container():
        st.markdown("<div class='map-card'>", unsafe_allow_html=True)
        control_cols = st.columns([2, 1])
        with control_cols[0]:
            frame_idx = st.slider(
                "Frise horaire (heures)",
                min_value=min_frame,
                max_value=max_frame,
                value=st.session_state.get("map_frame_idx", min_frame),
                step=1,
                key="map_frame_slider",
                format="T+%d h",
            )
            st.session_state["map_frame_idx"] = frame_idx
            if st.button("▶️ Lancer l'animation", key="map_play_button"):
                placeholder = st.empty()
                for next_idx in list(range(frame_idx, max_frame + 1)) + list(range(min_frame, frame_idx)):
                    st.session_state["map_frame_idx"] = next_idx
                    frame_idx = next_idx
                    placeholder.write(f"Animation T+{next_idx} h")
                    time.sleep(0.25)
        with control_cols[1]:
            granularity = st.radio(
                "Pastilles",
                ["Région", "Département"],
                index=0,
                horizontal=False,
                key="map_granularity",
            )

        layer_labels = [CLIMATE_VARIABLES[key].get("label", key) for key in CLIMATE_VARIABLES]
        label_to_key = {CLIMATE_VARIABLES[key].get("label", key): key for key in CLIMATE_VARIABLES}
        default_label = CLIMATE_VARIABLES["Température"].get("label", "Température")
        with st.expander("Calques météo"):
            layer_cols = st.columns([2, 2, 1])
            with layer_cols[0]:
                heat_selection = st.multiselect(
                    "Heatmaps",
                    options=layer_labels,
                    default=[default_label],
                    key="map_heat_layers",
                )
            with layer_cols[1]:
                marker_selection = st.multiselect(
                    "Pastilles régionales",
                    options=layer_labels,
                    default=[default_label],
                    key="map_marker_layers",
                )
            with layer_cols[2]:
                show_providers = st.checkbox("Fournisseurs", value=False, key="map_show_providers")
                show_wind_vectors = st.checkbox("Vecteurs vent", value=False, key="map_show_wind_vectors")

        heat_layers = [label_to_key[label] for label in heat_selection if label in label_to_key]
        marker_layers = [label_to_key[label] for label in marker_selection if label in label_to_key]
        if not heat_layers:
            heat_layers = ["Température"]
        if not marker_layers:
            marker_layers = ["Température"]
        provider_layer = show_providers
        wind_vector_layer = show_wind_vectors

        df_frame = df_climat[df_climat["frame"] == frame_idx]
        if df_frame.empty:
            st.warning("Aucune donnée pour cette échéance.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        ts_label = f"T+{frame_idx} h"
        if timeline is not None and frame_idx < len(timeline):
            try:
                ts = pd.to_datetime(timeline[frame_idx])
                ts_label = ts.strftime("%a %d %b · %Hh").capitalize()
            except Exception:
                pass

    aggregated = aggregate_climate(df_frame, granularity)
    if not aggregated.empty:
        aggregated = aggregated.sort_values("pop", ascending=False).head(70)

    density_source = df_frame.sort_values("pop", ascending=False).head(800)
    if user_position:
        center = user_position
    elif not aggregated.empty:
        center = (float(aggregated["lat"].mean()), float(aggregated["lon"].mean()))
    else:
        center = (46.2276, 2.2137)

    fig = render_base_map(granularity, center)
    for idx, variable in enumerate(heat_layers):
        render_heatmap_layer(
            fig,
            density_source,
            variable,
            granularity,
            visible=True,
            show_colorbar=(idx == 0),
        )

    for variable in CLIMATE_VARIABLES.keys():
        render_metric_pills(
            fig,
            aggregated,
            variable,
            visible=variable in marker_layers,
            granularity=granularity,
        )
    render_fournisseur_layer(fig, df_fournisseurs, visible=provider_layer)
    render_wind_vector_layer(
        fig,
        aggregated,
        visible=wind_vector_layer,
        granularity=granularity,
    )

    if user_position:
        fig.add_trace(
            go.Scattermapbox(
                lat=[user_position[0]],
                lon=[user_position[1]],
                mode="markers",
                marker=dict(size=18, color="#16a34a", opacity=0.95, symbol="star"),
                name="Ma position",
            )
        )

    fig.update_layout(
        legend=dict(
            title="Calques",
            x=0.99,
            xanchor="right",
            y=0.99,
            bgcolor="rgba(248,250,252,0.85)",
            bordercolor="#e2e8f0",
            borderwidth=1,
        ),
    )
    fig.add_annotation(
        text=f"Projection : {ts_label}",
        x=0.01,
        y=0.01,
        xref="paper",
        yref="paper",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#cbd5f5",
        borderwidth=1,
        font=dict(color="#0f172a"),
        showarrow=False,
        align="left",
    )

    st.markdown(
        f"""
        <div class='header'>
            <strong>Exploration multi-couches</strong>
            <span>{ts_label} · {granularity}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

def render_map_tab(cfg, map_engine: MapEngine):
    communes = map_engine.load_communes()
    df_climat, timeline = build_climate_snapshot(communes, frames=24)
    df_fournisseurs = load_fournisseurs_data()
    user_position = (cfg.latitude, cfg.longitude) if cfg.latitude and cfg.longitude else None
    render_france_map(df_climat, df_fournisseurs, timeline, user_position=user_position)



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
    vacances_profile = STOVE_MODES.get("Vacances")
    if stove_mode == "Vacances" and vacances_profile and vacances_profile.get("pulse_target_temp") is not None:
        target_temp_effective = min(target_temp_effective, float(vacances_profile["pulse_target_temp"]))
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
    plan_horizon = heating_plan["horizon"] if heating_plan["horizon"] > 0 else min(PLANNING_HORIZON_HOURS, len(df) or PLANNING_HORIZON_HOURS)
    if plan_horizon <= 0:
        plan_horizon = 24
    active_mask = (
        heating_plan["active_mask"]
        if heating_plan["horizon"] > 0
        else [i < hours_on for i in range(plan_horizon)]
    )
    effective_hours_on = heating_plan["active_hours"] if heating_plan["horizon"] > 0 else hours_on
    pellet_hours_for_rate = max(1, int(effective_hours_on))
    effective_hours_for_kpis = max(0, int(effective_hours_on))
    pellet_engine = PelletEngine(cfg.power_kw, cfg.efficiency, cfg.pellet_price_bag)
    pellet_df = pellet_engine.compute_pellet_usage(
        hours=plan_horizon,
        target_temp=target_temp_effective,
        hours_on=pellet_hours_for_rate,
        desired_duration_hours=DEFAULT_BAG_DURATION_HOURS,
        active_mask=active_mask,
    )
    pellet_usage = summarize_pellet_usage(pellet_df, plan_horizon)

    cost, deg, loss, pellet_cost, electric_cost, meteo_stats = compute_kpis(
        cfg,
        df,
        effective_hours_for_kpis,
        electric_price,
        pellet_df,
    )

    trigger_ts, reach_ts = derive_schedule_markers(heating_plan, cfg, model, target_temp_effective)
    area_m2 = cfg.volume_m3 / 2.5 if cfg.volume_m3 else 0.0
    temp_ext_value = (
        float(df["temp_ext"].iloc[0])
        if not df.empty and "temp_ext" in df
        else float(cfg.temp_current)
    )
    ui_temp_target = float(st.session_state.get("ui_temp_target", target_temp_effective))
    ui_temp_current = float(st.session_state.get("ui_temp_current", cfg.temp_current))
    ramp_cap = heating_plan.get("max_ramp_c_per_hour") or STOVE_MODES.get(stove_mode, {}).get("max_delta_per_hour", STANDARD_HOME_SETTINGS["max_delta_per_hour"])
    wind_series = heating_plan.get("wind_series") if isinstance(heating_plan, dict) else None
    wind_reference = None
    if isinstance(wind_series, pd.Series) and not wind_series.empty:
        wind_reference = float(wind_series.iloc[0])
    elif not df.empty and "wind" in df:
        wind_reference = float(df["wind"].iloc[0])
    else:
        wind_reference = 0.0
    eta_hours_live = None
    try:
        effective_ext = temp_ext_value - 0.2 * float(wind_reference)
        eta_raw = model.time_to_reach(
            ui_temp_current,
            ui_temp_target,
            effective_ext,
            max_delta_per_hour=ramp_cap,
        )
        eta_hours_live = enforce_realistic_eta(ui_temp_current, ui_temp_target, eta_raw)
    except Exception:
        eta_hours_live = None
    eta_source_label = "Modèle thermique personnalisé" if eta_hours_live is not None else "Scénario logement standard 100 m²"

    now_reference = pd.Timestamp.now(tz="Europe/Paris")
    trigger_label = now_reference.isoformat()
    reach_label = "--"
    if eta_hours_live is not None:
        reach_label = (now_reference + pd.Timedelta(hours=float(eta_hours_live))).isoformat()
    elif reach_ts is not None:
        reach_label = reach_ts.isoformat()

    dashboard_payload = {
        "temp_ext": temp_ext_value,
        "temp_target": ui_temp_target,
        "temp_current": ui_temp_current,
        "surface_label": f"{area_m2:.0f} m²",
        "isolation_label": cfg.isolation,
        "heure_declenchement": trigger_label,
        "heure_objectif": reach_label,
        "pellets_48h": pellet_usage.get("total_bags", 0.0),
        "cost_pellets": pellet_cost,
        "cost_elec": electric_cost,
        "cost_total": cost,
        "temp_bounds": (16.0, 25.0),
        "eta_hours": eta_hours_live,
        "eta_source_label": eta_source_label,
    }
    render_green_dashboard(dashboard_payload)

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
        render_weather_tab(df, pellet_df, hours_on, heating_plan)

    with tab3:
        render_kpi_tab(cost, deg, loss, pellet_cost, electric_cost, meteo_stats, pellet_usage)

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
