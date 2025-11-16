import math
import streamlit as st
import pandas as pd
import plotly.express as px
from core.user_inputs import UserConfig
from data.data_meteo import MeteoClient
from models.thermal_model import ThermalModel
from core.kpi_engine import KPIEngine
from core.diagnostic import Diagnostic
from data.data_france import FranceMeteo
import plotly.graph_objects as go


def sidebar_inputs():
    st.sidebar.markdown("## Paramètres")
    st.sidebar.info("Ajustez les paramètres de votre habitation pour simuler le comportement thermique.")
    volume = st.sidebar.number_input("Volume chauffé (m3)", min_value=20.0, max_value=1000.0, value=250.0)
    inertia = st.sidebar.selectbox("Inertie thermique", ["faible", "moyenne", "forte"])
    power = st.sidebar.number_input("Puissance poêle (kW)", min_value=1.0, max_value=20.0, value=8.0)
    efficiency = st.sidebar.slider("Rendement", min_value=0.5, max_value=1.0, value=0.85)
    temp_current = st.sidebar.number_input("Température intérieure actuelle", value=17.0)
    temp_target = st.sidebar.number_input("Température intérieure cible", value=20.0)
    latitude = st.sidebar.number_input("Latitude", value=48.8)
    longitude = st.sidebar.number_input("Longitude", value=2.3)
    isolation = st.sidebar.selectbox("Isolation", ["faible", "moyenne", "forte"])
    price = st.sidebar.number_input("Prix de l'énergie (€/kWh)", value=0.18)
    hours_on = st.sidebar.slider("Durée de chauffe prévue (h)", 0, 24, 4)

    cfg = UserConfig(
        volume_m3=volume,
        inertia_level=inertia,
        power_kw=power,
        efficiency=efficiency,
        temp_current=temp_current,
        temp_target=temp_target,
        latitude=latitude,
        longitude=longitude,
        isolation=isolation
    )

    return cfg, price, hours_on

def load_meteo(cfg):
    client = MeteoClient(cfg.latitude, cfg.longitude)
    df = client.fetch_hourly()
    return df

def compute_projection(cfg, df, hours_on):
    model = ThermalModel(
        tau_hours=cfg.tau_hours(),
        volume_m3=cfg.volume_m3,
        power_kw=cfg.power_kw,
        efficiency=cfg.efficiency
    )
    T_int_start = cfg.temp_current
    ext_series = df["temp_ext"].values[:24]
    wind_series = df["wind"].values[:24]
    proj = model.simulate(T_int_start, ext_series, wind_series, hours_on)
    return proj

def compute_kpis(cfg, df, hours_on, price):
    kpi = KPIEngine(price)
    cost = kpi.daily_cost(hours_on, cfg.power_kw)
    deg = kpi.degree_day_ratio(df)
    loss = kpi.thermal_loss(cfg.volume_m3, cfg.tau_hours())
    return cost, deg, loss

def main():
    st.title("Thermo-Stats")

    cfg, price, hours_on = sidebar_inputs()

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
    proj = compute_projection(cfg, df, hours_on)
    cost, deg, loss = compute_kpis(cfg, df, hours_on, price)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Météo", "Thermique", "KPIs", "Diagnostic", "Carte France"])

    with tab1:
        st.subheader("Prévisions météo")
        st.write("Données météo horaires fournies par Météo-France (modèle Open-Meteo).")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Température extérieure (°C)", df["temp_ext"].iloc[0])
        with col2:
            st.metric("Vent (km/h)", df["wind"].iloc[0])

        st.divider()

        horizon = 24
        hours_axis = df["time"].dt.strftime("%H:%M").head(horizon)

        indoor_projection = proj.reindex(range(horizon)).astype(float)
        indoor_projection.iloc[14:] = float("nan")

        pellet_energy_kwh_per_kg = 4.8
        pellet_bags_per_hour = 0.0
        if cfg.efficiency > 0:
            pellet_bags_per_hour = (cfg.power_kw / (pellet_energy_kwh_per_kg * cfg.efficiency)) / 15
        pellet_curve = [pellet_bags_per_hour if i < hours_on else 0 for i in range(horizon)]

        meteo_plot = pd.DataFrame(
            {
                "Heure (24h)": hours_axis,
                "Température extérieure (°C)": df["temp_ext"].head(horizon).values,
                "Vent (km/h)": df["wind"].head(horizon).values,
                "Température intérieure projetée (°C)": indoor_projection.values,
                "Consommation pellet (sac de 15 kg / h)": pellet_curve,
            }
        )

        meteo_long = meteo_plot.melt(
            id_vars="Heure (24h)", var_name="Mesure", value_name="Valeur"
        )

        fig_meteo = px.line(
            meteo_long,
            x="Heure (24h)",
            y="Valeur",
            color="Mesure",
            markers=True,
            title="Prévisions sur 24h (température, vent, pellet et intérieur)"
        )
        fig_meteo.update_layout(
            xaxis_title="Heure (format 24h)",
            yaxis_title="Valeur (unités mixtes)",
            legend_title="Courbe",
        )

        st.plotly_chart(fig_meteo, use_container_width=True)

    with tab2:
        st.write("Simulation thermique basée sur l'inertie de l'habitation et la météo horaire.")
        st.divider()

        st.subheader("Heure d'allumage recommandée")

        model = ThermalModel(
            tau_hours=cfg.tau_hours(),
            volume_m3=cfg.volume_m3,
            power_kw=cfg.power_kw,
            efficiency=cfg.efficiency
        )

        with st.expander("Modèle RC (scientifique)"):
            col_tau, col_c, col_r = st.columns(3)
            col_tau.metric("Constante de temps (τ)", f"{cfg.tau_hours():.1f} h")
            col_c.metric("Capacité thermique (C)", f"{model.heat_capacity():.1f} kWh/K")
            resistance = model.thermal_resistance()
            if math.isinf(resistance):
                r_value = "∞"
            else:
                r_value = f"{resistance:.2f} h·K/kWh"
            col_r.metric("Résistance thermique (R)", r_value)
            st.latex(r"\frac{dT}{dt} = \frac{T_{ext}-T_{int}}{\tau} + \frac{P \eta}{C}")
            st.caption("Modèle 1R-1C utilisé pour la simulation numérique de l'habitation.")

        T_eff_now = df["temp_ext"].iloc[0] - 0.2 * df["wind"].iloc[0]

        dt = model.time_to_reach(cfg.temp_current, cfg.temp_target, T_eff_now)

        fire_time = None
        time_now = pd.Timestamp.now()

        if dt is None:
            st.warning("Impossible d'atteindre la température cible avec les paramètres actuels.")
        else:
            fire_time = time_now
            target_eta = time_now + pd.Timedelta(hours=dt)
            st.metric("Temps nécessaire pour atteindre la température cible (h)", round(dt, 2))
            st.markdown(
                f"""
                <div style="padding: 14px; border-radius: 6px; background-color: #0275d8; color: white; font-weight: bold;">
                    Allumer le poêle à : {fire_time.strftime('%H:%M')} — ETA {target_eta.strftime('%H:%M')}
                </div>
                """,
                unsafe_allow_html=True
            )

            timeline = pd.DataFrame(
                [
                    {
                        "Phase": "Allumage",
                        "Début": time_now,
                        "Fin": time_now + pd.Timedelta(minutes=5),
                    },
                    {
                        "Phase": "Montée jusqu'à consigne",
                        "Début": time_now + pd.Timedelta(minutes=5),
                        "Fin": target_eta,
                    },
                    {
                        "Phase": "Maintien estimé",
                        "Début": target_eta,
                        "Fin": target_eta + pd.Timedelta(hours=max(1, hours_on - dt)),
                    },
                ]
            )
            fig_eta = px.timeline(
                timeline,
                x_start="Début",
                x_end="Fin",
                y="Phase",
                color="Phase"
            )
            fig_eta.update_layout(
                xaxis_title="Horloge (24h)", yaxis_title="Étapes", showlegend=False
            )
            st.plotly_chart(fig_eta, use_container_width=True)

        st.write("")
        st.subheader("Alerte d’allumage")

        if fire_time is None:
            st.info("Aucune alerte disponible.")
        else:
            delta = (fire_time - time_now).total_seconds() / 60
            if delta <= 15:
                st.error("Allumer le poêle maintenant.")
            elif delta <= 45:
                st.warning(f"Allumer le poêle dans environ {int(delta)} minutes.")
            else:
                st.info("Aucune alerte imminente.")

        st.write("")
        st.subheader("Température cible")

        st.metric("Consigne", f"{cfg.temp_target} °C")

        T_ext_arr = df["temp_ext"].iloc[:24].to_numpy(dtype=float)
        wind_arr = df["wind"].iloc[:24].to_numpy(dtype=float)
        T_eff_series = T_ext_arr - 0.2 * wind_arr

        temps_cible = model.time_series_until_target(cfg.temp_current, cfg.temp_target, T_eff_series)

        if len(temps_cible) == 0:
            st.warning("La montée en température ne permet pas d'atteindre la cible dans la fenêtre horaire.")
        else:
            st.line_chart(pd.Series(temps_cible, name="Montée jusqu’à consigne"))

        st.write("")
        st.subheader("Météo horaire utilisée")
        st.dataframe(df[["time", "temp_ext", "wind"]].head(24))


    with tab3:
        st.write("Indicateurs calculés automatiquement à partir des données horaires et des paramètres de l'habitation.")
        st.divider()
        st.subheader("Indicateurs clés")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Coût journalier estimé (€)", round(cost, 2))
        with col2:
            st.metric("Degré-jours", deg if deg is not None else "N/A")
        with col3:
            st.metric("Déperdition thermique (W/K)", round(loss, 2))

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
            unsafe_allow_html=True
        )

        with st.expander("Légendes (définitions)"):
            st.markdown(
                "- **Coût journalier estimé** : consommation horaire du poêle multipliée par le prix de l'énergie.\n"
                "- **Degré-jours** : indicateur d'écart moyen entre la température intérieure de consigne et la température extérieure.\n"
                "- **Déperdition thermique** : puissance perdue par degré d'écart (W/K), reflet de l'isolation et des fuites."
            )


    with tab4:
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
            unsafe_allow_html=True
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


    with tab5:
        st.write("Carte interactive des températures en France.")
        st.divider()

        st.subheader("Carte thermique de la France")

        fm = FranceMeteo()
        df_fr = fm.fetch()

        layer_options = {
            "Température extérieure": ("temp", "RdBu_r", "°C"),
            "Humidité relative": ("humidity", "Blues", "%"),
            "Vent": ("wind", "PuBu", "km/h"),
            "Température ressentie": ("apparent", "OrRd", "°C"),
        }

        selected_layers = st.multiselect(
            "Couches à afficher (gradients)", list(layer_options.keys()), default=["Température extérieure"]
        )

        if not selected_layers:
            selected_layers = ["Température extérieure"]

        fig = go.Figure()

        for layer in selected_layers:
            col, scale, unit = layer_options[layer]
            fig.add_trace(
                go.Scattermapbox(
                    lat=df_fr["lat"],
                    lon=df_fr["lon"],
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=df_fr[col],
                        colorscale=scale,
                        showscale=True,
                        colorbar=dict(title=f"{layer} ({unit})"),
                        opacity=0.8,
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        f"{layer}: %{{marker.color:.1f}} {unit}<br>"
                        "Vent: %{customdata[1]} km/h<br>"
                        "Humidité: %{customdata[2]}%<br>"
                        "Conseil: %{customdata[3]}<extra></extra>"
                    ),
                    customdata=df_fr[["city", "wind", "humidity", "start_hint"]].values,
                    name=layer,
                )
            )

        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_zoom=4,
            mapbox_center={"lat": 46.5, "lon": 2.5},
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            height=650,
            legend_title="Calques actifs",
        )

        st.caption("Pastilles redimensionnées selon l'échelle de carte et colorées par couches météorologiques sélectionnées.")
        st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    <hr>
    <div style="text-align: center; color: gray;">
        Thermo-Stats — Prototype énergétique © 2025
    </div>
    """,
    unsafe_allow_html=True
)

if __name__ == "__main__":
    main()
