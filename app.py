import math
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from core.user_inputs import UserConfig, STRUCTURE_PRESETS
from data.data_meteo import MeteoClient
from models.thermal_model import ThermalModel
from core.kpi_engine import KPIEngine
from core.diagnostic import Diagnostic
from core.map_engine import MapEngine
from core.pellet_engine import PelletEngine
from data.data_france import FranceMeteo


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
    structure = st.sidebar.selectbox("Structure / paroi", UserConfig.structures(), index=1,
                                     help="Détermine automatiquement C et G (inertie et pertes)")
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
        "Isolation", ["faible", "moyenne", "forte"],
        format_func=lambda x: {
            "faible": "Années 50–80, simple vitrage, ponts thermiques",
            "moyenne": "Années 90–2010, double vitrage",
            "forte": "RT2012 ou rénovation complète",
        }.get(x, x),
    )

    st.sidebar.markdown("### Poêle")
    pellet_price = st.sidebar.number_input("Prix du sac de pellets (15 kg)", min_value=3.0, max_value=20.0, value=7.5)
    hours_on = st.sidebar.slider("Durée de chauffe prévue (h)", 0, 24, 6)

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
    )

    return cfg, pellet_price, hours_on, map_engine


def load_meteo(cfg):
    client = MeteoClient(cfg.latitude, cfg.longitude)
    df = client.fetch_hourly()
    return df


def compute_projection(cfg, df, hours_on):
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
    proj = model.simulate(T_int_start, ext_series, wind_series, hours_on)
    return proj, model


def compute_kpis(cfg, df, hours_on, price, pellet_df):
    kpi = KPIEngine(price)
    cost = kpi.daily_cost(hours_on, cfg.power_kw)
    deg = kpi.degree_day_ratio(df)
    loss = kpi.thermal_loss(cfg.volume_m3, cfg.tau_hours())
    pellet_cost = kpi.pellet_cost(pellet_df)
    electric_cost = kpi.stove_electric_cost(hours_on)
    return cost, deg, loss, pellet_cost, electric_cost


def render_weather_tab(df, indoor_proj, pellet_df, hours_on):
    st.subheader("Prévisions météo (24h)")
    st.write("Données météo horaires fournies par Météo-France (Open-Meteo).")
    horizon = 24
    hours_axis = df["time"].dt.strftime("%H:%M").head(horizon)

    indoor_projection = indoor_proj.reindex(range(horizon)).astype(float)
    indoor_projection.iloc[14:] = float("nan")

    pellet_curve = pellet_df["bags_used"].reindex(range(horizon)).fillna(0)
    pellet_curve = pellet_curve.where(~pellet_df["recharge"].reindex(range(horizon)).fillna(False))

    meteo_plot = pd.DataFrame(
        {
            "Heure (24h)": hours_axis,
            "Température extérieure (°C)": df["temp_ext"].head(horizon).values,
            "Température intérieure projetée (°C)": indoor_projection.values,
            "Vent (km/h)": df["wind"].head(horizon).values,
            "Consommation pellet (sac/h)": pellet_curve.values,
        }
    )
    meteo_long = meteo_plot.melt(id_vars="Heure (24h)", var_name="Mesure", value_name="Valeur")

    fig_meteo = px.line(
        meteo_long,
        x="Heure (24h)",
        y="Valeur",
        color="Mesure",
        markers=True,
        title="Prévisions sur 24h (températures, vent et pellet)",
    )
    fig_meteo.update_layout(
        xaxis_title="Heure (format 24h)",
        yaxis_title="Valeur (unités mixtes)",
        legend_title="Courbe",
    )
    recharge_hours = [hours_axis.iloc[i] for i in pellet_df.index[pellet_df["recharge"]]]
    if recharge_hours:
        fig_meteo.add_trace(
            go.Scatter(
                x=recharge_hours,
                y=[pellet_curve.max()] * len(recharge_hours),
                mode="markers",
                marker_symbol="triangle-up",
                marker_color="#e67e22",
                name="Point de recharge",
                showlegend=True,
            )
        )
    st.plotly_chart(fig_meteo, use_container_width=True)



def render_thermal_tab(cfg, df, model, indoor_proj, pellet_df, hours_on):
    st.write("Simulation thermique basée sur l'inertie de l'habitation et la météo horaire.")
    horizon = 24
    hours_axis = df["time"].dt.strftime("%H:%M").head(horizon).tolist()

    T_ext_arr = df["temp_ext"].iloc[:horizon].to_numpy(dtype=float)
    wind_arr = df["wind"].iloc[:horizon].to_numpy(dtype=float)
    T_eff_series = T_ext_arr - 0.2 * wind_arr

    timeline_df = pd.DataFrame(
        {
            "Heure": hours_axis,
            "Température extérieure": df["temp_ext"].head(horizon).values,
            "Vent": df["wind"].head(horizon).values,
            "HR": df["humidity"].head(horizon).values if "humidity" in df else 0,
            "Température intérieure projetée": indoor_proj.head(horizon).values,
            "Pellets sac/h": pellet_df["bags_used"].head(horizon).values,
        }
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timeline_df["Heure"], y=timeline_df["Température intérieure projetée"],
                             name="Intérieure projetée (°C)", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=timeline_df["Heure"], y=[cfg.temp_target]*horizon,
                             name="Consigne", mode="lines", line=dict(dash="dash", color="#2ca02c")))
    fig.add_trace(go.Scatter(x=timeline_df["Heure"], y=timeline_df["Température extérieure"],
                             name="Température extérieure (°C)", mode="lines"))
    fig.add_trace(go.Scatter(x=timeline_df["Heure"], y=timeline_df["Vent"],
                             name="Vent (km/h)", mode="lines", yaxis="y2"))
    fig.add_trace(go.Bar(x=timeline_df["Heure"], y=timeline_df["Pellets sac/h"],
                         name="Pellets (sac/h)", yaxis="y3", marker_color="#e67e22", opacity=0.6))
    recharge_hours = [hours_axis[i] for i in pellet_df.index[pellet_df["recharge"]]]
    if recharge_hours:
        fig.add_trace(
            go.Scatter(
                x=recharge_hours,
                y=[timeline_df["Pellets sac/h"].max() * 1.05] * len(recharge_hours),
                mode="markers+text",
                text=["Recharge"] * len(recharge_hours),
                textposition="top center",
                marker_symbol="triangle-up",
                marker_color="#e67e22",
                name="Points de recharge",
                yaxis="y3",
            )
        )

    fig.update_layout(
        xaxis=dict(domain=[0, 0.9]),
        yaxis=dict(title="Température (°C)"),
        yaxis2=dict(title="Vent (km/h)", overlaying="y", side="right", showgrid=False),
        yaxis3=dict(title="Pellets (sac/h)", anchor="x", overlaying="y", side="right", position=0.92,
                    showgrid=False, range=[0, max(0.2, timeline_df["Pellets sac/h"].max()*1.3)]),
        legend_title="Courbes",
        title="Fresque temporelle unifiée (24h)",
        bargap=0.05,
    )
    fig = apply_timeline_styling_day_night(fig, hours_axis)

    hatched = dict(
        type="rect",
        xref="x",
        yref="paper",
        x0=hours_axis[0],
        x1=hours_axis[min(hours_on, len(hours_axis)-1)],
        y0=0,
        y1=1,
        fillcolor="rgba(255,165,0,0.15)",
        line=dict(color="rgba(255,140,0,0.4)", dash="dot"),
        layer="below",
    )
    fig.add_shape(hatched)
    fig.add_annotation(x=hours_axis[min(hours_on, len(hours_axis)-1)], y=cfg.temp_target,
                       text="Attention – allumage", showarrow=True, arrowhead=2,
                       bgcolor="rgba(255,165,0,0.2)")

    st.plotly_chart(fig, use_container_width=True)



def render_kpi_tab(cost, deg, loss, pellet_cost, electric_cost):
    st.write("Indicateurs calculés automatiquement à partir des données horaires et des paramètres de l'habitation.")
    st.divider()
    st.subheader("Indicateurs clés")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Coût journalier estimé (€)", round(cost, 2), help="Consommation totale du poêle multipliée par le prix de l'énergie.")
    with col2:
        st.metric("Degré-jours", deg if deg is not None else "N/A", help="Différence cumulée entre consigne (18°C) et température extérieure sur la période.")
    with col3:
        st.metric("Déperdition thermique (W/K)", round(loss, 2), help="Puissance perdue pour chaque degré d'écart entre intérieur et extérieur.")

    st.markdown(
        f"Coût pellet: **{pellet_cost:.2f} €** — Coût élec du poêle: **{electric_cost:.2f} €**",
        help="Coût énergique journalier = consommation pellets + électricité du poêle",
    )

    if loss < 80:
        color = "#5CB85C"
        txt = "faible"
    elif loss < 150:
        color = "#F0AD4E"
        txt = "moyenne"
    else:
        color = "#D9534F"

    st.markdown(
        f"""
        <div style="padding: 12px; border-radius: 8px; background-color: {color}; color: white;">
            Niveau de pertes thermiques : <b>{txt}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_diagnostic_tab(cfg, loss):
    st.write("Évaluation du comportement thermique de l'habitation selon les pertes estimées.")
    st.divider()
    st.subheader("Diagnostic thermique")

    diag = Diagnostic(loss, cfg.isolation)
    result = diag.summary()

    cls = result["classe"]
    score = result["score"]
    exp = result["explication"]
    rec = result["recommandation"]

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

    base_fig = map_engine.base_figure(center_lat=cfg.latitude, center_lon=cfg.longitude, zoom=zoom)

    size_hint = df_fr["pop"].fillna(0).apply(lambda p: max(6, min(18, (p + 1) ** 0.25))) if "pop" in df_fr else None

    for layer in selected_layers:
        col, scale, unit = layer_options[layer]
        base_fig.add_trace(map_engine.build_layer(df_fr, layer, df_fr[f"{col}_0"], scale, unit, sizes=size_hint))

    frames = map_engine.calc_timelapse_frames(
        df_fr,
        timeline,
        {name: {"scale": layer_options[name][1], "unit": layer_options[name][2]} for name in selected_layers},
    )
    base_fig.frames = frames
    map_engine.add_timelapse_controls(base_fig)
    map_engine.pick_on_map(base_fig, cfg.latitude, cfg.longitude)

    st.caption("Pastilles redimensionnées selon l'échelle de carte et colorées par couches météorologiques sélectionnées.")
    st.plotly_chart(base_fig, use_container_width=True)



def main():
    st.title("Thermo-Stats")

    cfg, price, hours_on, map_engine = sidebar_inputs()

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
    proj, model = compute_projection(cfg, df, hours_on)
    pellet_engine = PelletEngine(cfg.power_kw, cfg.efficiency, cfg.pellet_price_bag)
    pellet_df = pellet_engine.compute_pellet_usage(hours=24, target_temp=cfg.temp_target, active_mask=[i < hours_on for i in range(24)])

    cost, deg, loss, pellet_cost, electric_cost = compute_kpis(cfg, df, hours_on, price, pellet_df)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Météo", "Thermique", "KPIs", "Diagnostic", "Carte France"])

    with tab1:
        render_weather_tab(df, proj, pellet_df, hours_on)

    with tab2:
        render_thermal_tab(cfg, df, model, proj, pellet_df, hours_on)

    with tab3:
        render_kpi_tab(cost, deg, loss, pellet_cost, electric_cost)

    with tab4:
        render_diagnostic_tab(cfg, loss)

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
