import streamlit as st
import pandas as pd

from core.user_inputs import UserConfig
from data.data_meteo import MeteoClient
from models.thermal_model import ThermalModel
from core.kpi_engine import KPIEngine
from core.diagnostic import Diagnostic


def sidebar_inputs():
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

    df = load_meteo(cfg)
    proj = compute_projection(cfg, df, hours_on)
    cost, deg, loss = compute_kpis(cfg, df, hours_on, price)

    tab1, tab2, tab3, tab4 = st.tabs(["Météo", "Thermique", "KPIs", "Diagnostic"])

    with tab1:
        st.subheader("Prévisions météo")
        st.line_chart(df[["temp_ext", "wind"]])

    with tab2:
        st.subheader("Projection température intérieure")
        st.line_chart(proj)

    with tab3:
        st.subheader("Indicateurs")
        st.write("Coût journalier estimé (€):", round(cost, 2))
        st.write("Degré-jours:", deg)
        st.write("Déperdition estimée (W/K):", round(loss, 2))

    with tab4:
        st.subheader("Diagnostic de l'habitation")
        st.write("En cours d'implémentation.")

if __name__ == "__main__":
    main()
