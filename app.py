import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core.user_inputs import UserConfig, STRUCTURE_PRESETS
from data.data_meteo import MeteoClient
from models.thermal_model import ThermalModel
from core.kpi_engine import KPIEngine
from core.diagnostic import Diagnostic
from core.map_engine import MapEngine
from core.pellet_engine import PelletEngine
from data.data_france import FranceMeteo


DEFAULT_BAG_DURATION_HOURS = 18.0


def adjusted_target_temperature(temp_target: float, temp_current: float, desired_duration_hours: float) -> float:
    """Ajuste légèrement la consigne pour refléter la durée souhaitée d'un sac."""
    if desired_duration_hours <= 0:
        desired_duration_hours = DEFAULT_BAG_DURATION_HOURS
    baseline = DEFAULT_BAG_DURATION_HOURS
    span = max(1.0, baseline)
    delta = (baseline - desired_duration_hours) / span * 2.5
    adjusted = temp_target + delta
    adjusted = min(temp_target + 3.0, adjusted)
    adjusted = max(temp_current + 0.3, adjusted)
    return round(adjusted, 1)


def apply_timeline_styling_day_night(fig: go.Figure, hours: list[str]):
    """Ajoute un fond jour/nuit et un curseur d'heure courante sur un graphe Plotly."""
    shapes = []
    night_blocks = [(0, 6), (20, 24)]
    for start, end in night_blocks:
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=hours[start],
                x1=hours[min(end, len(hours) - 1)],
                y0=0,
                y1=1,
                fillcolor="rgba(20,20,60,0.08)",
                line=dict(width=0),
                layer="below",
            )
        )
    current_hour = pd.Timestamp.now().hour
    current_label = hours[min(current_hour, len(hours) - 1)]
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


def sidebar_inputs():
    st.sidebar.markdown("## Paramètres")
    st.sidebar.info("Ajustez les paramètres de votre habitation pour simuler le comportement thermique.")
    volume = st.sidebar.number_input("Volume chauffé (m³)", min_value=20.0, max_value=1200.0, value=250.0)
    inertia = st.sidebar.selectbox("Inertie thermique (label)", ["faible", "moyenne", "forte"], index=1)
    structure = st.sidebar.selectbox(
        "Structure / paroi",
        UserConfig.structures(),
        index=1,
        help="Familles structure/isolant générées automatiquement (C et G).",
    )
    glazing = st.sidebar.selectbox("Type de vitrage", UserConfig.glazing_choices(), index=1)
    vmc = st.sidebar.selectbox("VMC", UserConfig.vmc_choices(), help="Impacte les infiltrations d'air")
    power = st.sidebar.number_input("Puissance poêle (kW)", min_value=1.0, max_value=20.0, value=8.0)
    efficiency = st.sidebar.slider("Rendement", min_value=0.5, max_value=1.0, value=0.85)
    temp_current = st.sidebar.number_input("Température intérieure actuelle", value=17.0)
    temp_target = st.sidebar.number_input("Température intérieure cible", value=20.0)

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

    st.sidebar.markdown("### Isolation")
    isolation = st.sidebar.selectbox(
        "Isolation",
        options=["faible", "moyenne", "forte"],
        format_func=lambda x: str(
            {
                "faible": "Années 50–80, simple vitrage, ponts thermiques",
                "moyenne": "Années 90–2010, double vitrage",
                "forte": "RT2012 ou rénovation complète",
            }.get(x, x)
        ),
    )

    st.sidebar.markdown("### Poêle")
    pellet_price = st.sidebar.number_input("Prix du sac de pellets (15 kg)", min_value=3.0, max_value=20.0, value=4.5)
    electric_price = st.sidebar.number_input("Prix électricité (€/kWh)", min_value=0.05, max_value=1.0, value=0.18, step=0.01)
    hours_on = st.sidebar.slider("Durée de chauffe prévue (h)", 0, 24, 6)
    desired_consumption_hours = st.sidebar.slider(
        "Durée souhaitée pour 1 sac (h)",
        min_value=10,
        max_value=36,
        value=int(DEFAULT_BAG_DURATION_HOURS),
        help="Permet d'ajuster automatiquement la température cible pour économiser ou accélérer la combustion.",
    )

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

    return cfg, pellet_price, electric_price, hours_on, desired_consumption_hours, map_engine


def load_meteo(cfg):
    client = MeteoClient(cfg.latitude, cfg.longitude)
    df = client.fetch_hourly()
    return df


def compute_projection(cfg, df, hours_on, target_temp):
    model = ThermalModel(
        tau_hours=cfg.tau_hours(),
        volume_m3=cfg.volume_m3,
        power_kw=cfg.power_kw,
        efficiency=cfg.efficiency,
        infiltration_factor=cfg.loss_w_per_k() / max(0.001, STRUCTURE_PRESETS[cfg.structure]["loss_w_k"]),
        capacitance_kwh=cfg.capacitance_kwh(),
        loss_w_per_k=cfg.loss_w_per_k(),
    )
    T_int_start = cfg.temp_current
    ext_series = df["temp_ext"].values[:24]
    wind_series = df["wind"].values[:24]
    humidity_series = df["humidity"].values[:24] if "humidity" in df else None
    max_temp = max(target_temp + 5, cfg.temp_current + 10)
    proj = model.simulate(
        T_int_start,
        ext_series,
        wind_series,
        hours_on,
        humidity_series=humidity_series,
        target_temp=target_temp,
    ).clip(lower=5, upper=target_temp)
    return proj, model


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


def render_weather_tab(df, indoor_proj, pellet_df, hours_on):
    st.subheader("Sources météo et projections associées")
    horizon = min(24, len(df))
    if horizon == 0:
        st.warning("Impossible d'afficher les données météo.")
        return

    hours_axis = df["time"].dt.strftime("%H:%M").head(horizon).tolist()
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=hours_axis,
            y=df["temp_ext"].head(horizon).values,
            name="Température extérieure (°C)",
            mode="lines+markers",
            line=dict(color="#1f77b4"),
            marker=dict(color="#1f77b4"),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(
            x=hours_axis,
            y=df["wind"].head(horizon).values,
            name="Vent (km/h)",
            marker_color="#4fc3f7",
            opacity=0.5,
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
        fig.add_trace(
            go.Scatter(
                x=hours_axis,
                y=df["solar_radiation"].head(horizon).values,
                name="Rayonnement solaire (W/m²)",
                mode="lines",
                line=dict(color="#fdd835"),
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title="Fresque météo (24h)",
        legend_title="Séries",
        barmode="overlay",
    )
    fig.update_yaxes(title_text="Température (°C)", secondary_y=False)
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



def render_thermal_tab(cfg, df, model, indoor_proj, pellet_df, hours_on, target_temp, desired_consumption_hours):
    st.write("Simulation thermique basée sur l'inertie de l'habitation et la météo horaire.")
    delta = target_temp - cfg.temp_target
    if abs(delta) > 0.05:
        direction = "plus basse" if delta < 0 else "plus haute"
        st.caption(
            f"Consigne ajustée à {target_temp:.1f}°C (entrée {cfg.temp_target:.1f}°C) pour tenir environ {desired_consumption_hours} h par sac — consigne {direction}."
        )
    else:
        st.caption(
            f"Consigne maintenue à {target_temp:.1f}°C — durée cible d'un sac : {desired_consumption_hours} h."
        )

    horizon = min(24, len(df))
    if horizon == 0:
        st.warning("Données météo indisponibles pour la simulation thermique.")
        return
    hours_axis = df["time"].dt.strftime("%H:%M").head(horizon).tolist()

    indoor_series = pd.Series(indoor_proj, dtype="float64").reset_index(drop=True)
    indoor_series = indoor_series.reindex(range(horizon))
    if not indoor_series.empty:
        indoor_series.iloc[0] = float(cfg.temp_current)

    wind_series = df["wind"].head(horizon).astype(float).reset_index(drop=True)
    temp_ext_series = df["temp_ext"].head(horizon).astype(float).reset_index(drop=True)
    pellet_series = pellet_df["bags_used"].head(horizon).astype(float).reset_index(drop=True)

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

    timeline_df = pd.DataFrame({
        "Heure": hours_axis,
        "Température intérieure projetée": indoor_series.values,
        "Température extérieure": temp_ext_series.values,
        "Vent": wind_series.values,
        "Pellets sac/h": pellet_series.values,
    })
    if humidity_series is not None:
        timeline_df["Humidité relative"] = humidity_series.values
    if radiation_series is not None:
        timeline_df["Rayonnement solaire"] = radiation_series.values

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    custom_cols = []
    hover_template = "Heure: %{x}<br>Intérieure: %{y:.1f}°C"
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
            x=timeline_df["Heure"],
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
            x=timeline_df["Heure"],
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
            x=timeline_df["Heure"],
            y=timeline_df["Température extérieure"],
            name="Température extérieure (°C)",
            mode="lines+markers",
            line=dict(color="#1f77b4"),
            marker=dict(color="#1f77b4"),
            hovertemplate="Heure: %{x}<br>Extérieure: %{y:.1f}°C<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=timeline_df["Heure"],
            y=timeline_df["Vent"],
            name="Vent (km/h)",
            mode="lines",
            line=dict(color="#4fc3f7", dash="dot"),
            hovertemplate="Heure: %{x}<br>Vent: %{y:.0f} km/h<extra></extra>",
        ),
        secondary_y=True,
    )

    if "Humidité relative" in timeline_df:
        fig.add_trace(
            go.Scatter(
                x=timeline_df["Heure"],
                y=timeline_df["Humidité relative"],
                name="Humidité relative (%)",
                mode="lines",
                line=dict(color="#6a1b9a", dash="dashdot"),
                hovertemplate="Heure: %{x}<br>HR: %{y:.0f}%<extra></extra>",
            ),
            secondary_y=True,
        )

    if "Rayonnement solaire" in timeline_df:
        fig.add_trace(
            go.Scatter(
                x=timeline_df["Heure"],
                y=timeline_df["Rayonnement solaire"],
                name="Rayonnement solaire (W/m²)",
                mode="lines",
                line=dict(color="#fdd835"),
                hovertemplate="Heure: %{x}<br>Rayonnement: %{y:.0f} W/m²<extra></extra>",
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
    fig.update_yaxes(title_text="Température (°C)", secondary_y=False)
    fig.update_yaxes(title_text="Vent / HR / Rayonnement", secondary_y=True)

    fig = apply_timeline_styling_day_night(fig, hours_axis)

    start_idx = 0
    end_idx = max(0, min(hours_on - 1, len(hours_axis) - 1))
    hatched = dict(
        type="rect",
        xref="x",
        yref="paper",
        x0=hours_axis[start_idx],
        x1=hours_axis[end_idx],
        y0=0,
        y1=1,
        fillcolor="rgba(255,165,0,0.15)",
        line=dict(color="rgba(255,140,0,0.4)", dash="dot"),
        layer="below",
    )
    fig.add_shape(hatched)
    fig.add_annotation(
        x=hours_axis[start_idx],
        y=target_temp,
        text="Attention – allumage",
        showarrow=True,
        arrowhead=2,
        bgcolor="rgba(255,165,0,0.2)",
        xanchor="left",
    )

    if hours_on > 0 and len(hours_axis) > 0:
        vector_end_idx = min(hours_on - 1, len(hours_axis) - 1)
        fig.add_annotation(
            x=hours_axis[start_idx],
            y=float(cfg.temp_current),
            ax=hours_axis[vector_end_idx],
            ay=float(target_temp),
            axref="x",
            ayref="y",
            text="Influence du poêle",
            arrowcolor="#8d6e63",
            arrowwidth=2,
            arrowhead=3,
            bgcolor="rgba(141,110,99,0.15)",
        )

    st.plotly_chart(fig, use_container_width=True)



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

    with st.expander("Détails du diagnostic"):
        st.markdown(
            "- **Isolation saisie** : {isolation}\n"
            "- **Pertes estimées** : {loss:.1f} W/K — plus la valeur est élevée, plus la maison se refroidit vite.\n"
            "- **Inertie thermique** : {inertia} (impacte la vitesse de montée et de descente en température).\n"
            "- **Conseil** : surveiller les infiltrations d'air, les vitrages et l'isolation des combles pour améliorer le score."
            .format(isolation=cfg.isolation, loss=loss, inertia=cfg.inertia_level)
        )



def render_map_tab(cfg, map_engine: MapEngine):
    st.write("Carte interactive des températures en France.")
    st.divider()

    st.subheader("Carte thermique de la France")
    fm = FranceMeteo()

    zoom = st.slider("Zoom carte", min_value=3.0, max_value=8.0, value=4.5, step=0.5)
    communes_all = map_engine.load_communes()
    communes = map_engine.filter_by_zoom(communes_all, zoom)
    communes = communes.rename(columns={"nom": "city"}) if "nom" in communes.columns else communes

    weather = fm.fetch(communes)
    df_fr = weather["data"]
    timeline = list(weather["timeline"].strftime("%d/%m %Hh"))

    layer_options = {
        "Température": ("temp", "RdBu_r", "°C"),
        "Humidité": ("humidity", "Blues", "%"),
        "Vent": ("wind", "PuBu", "km/h"),
        "Ressentie": ("apparent", "OrRd", "°C"),
    }

    selected_layers = st.multiselect(
        "Couches à afficher (gradients)", list(layer_options.keys()), default=["Température"]
    )
    if not selected_layers:
        selected_layers = ["Température"]

    base_fig = map_engine.base_figure(center_lat=cfg.latitude, center_lon=cfg.longitude, zoom=int(round(zoom)))

    size_hint = df_fr["pop"].fillna(0).apply(lambda p: max(6, min(18, (p + 1) ** 0.25))) if "pop" in df_fr else None

    layers_meta = {
        name: {"scale": layer_options[name][1], "unit": layer_options[name][2], "col": layer_options[name][0]}
        for name in selected_layers
    }

    aggregates = {
        "Régions": {"df": map_engine.aggregate_layers(df_fr, layers_meta, by="region"), "label": "region"},
        "Départements": {"df": map_engine.aggregate_layers(df_fr, layers_meta, by="department"), "label": "department"},
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

    for level_name, meta_group in aggregates.items():
        agg_df = meta_group["df"]
        label_field = meta_group["label"]
        if agg_df.empty:
            continue
        for layer in selected_layers:
            meta = layers_meta[layer]
            col = meta["col"]
            base_fig.add_trace(
                map_engine.build_layer(
                    agg_df,
                    f"{layer} ({level_name})",
                    _pick_series(agg_df, col),
                    meta["scale"],
                    meta["unit"],
                    sizes=[14 if level_name == "Régions" else 11] * len(agg_df),
                    label_fields=[label_field],
                    show_scale=False,
                )
            )

    frames = map_engine.calc_timelapse_frames(
        df_fr,
        timeline,
        layers_meta,
        aggregates=aggregates,
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

    cfg, pellet_price, electric_price, hours_on, desired_consumption_hours, map_engine = sidebar_inputs()

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
    target_temp_effective = adjusted_target_temperature(
        cfg.temp_target, cfg.temp_current, desired_consumption_hours
    )
    proj, model = compute_projection(cfg, df, hours_on, target_temp_effective)
    pellet_engine = PelletEngine(cfg.power_kw, cfg.efficiency, cfg.pellet_price_bag)
    pellet_df = pellet_engine.compute_pellet_usage(
        hours=24,
        target_temp=target_temp_effective,
        hours_on=hours_on,
        desired_duration_hours=desired_consumption_hours,
        active_mask=[i < hours_on for i in range(24)],
    )

    cost, deg, loss, pellet_cost, electric_cost, meteo_stats = compute_kpis(cfg, df, hours_on, electric_price, pellet_df)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Météo", "Thermique", "KPIs", "Diagnostic", "Carte France"])

    with tab1:
        render_weather_tab(df, proj, pellet_df, hours_on)

    with tab2:
        render_thermal_tab(
            cfg,
            df,
            model,
            proj,
            pellet_df,
            hours_on,
            target_temp_effective,
            desired_consumption_hours,
        )

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
