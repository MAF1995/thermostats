import pandas as pd
from core.kpi_engine import KPIEngine

def test_daily_cost():
    kpi = KPIEngine(price_energy=0.18)
    pellet_df = pd.DataFrame({"cost_cum": [0.0, 2.7, 5.4]})
    cost = kpi.daily_cost(pellet_df=pellet_df, hours_on=4)
    assert cost > 5.4  # ajoute l'électricité de veille

def test_degree_day_ratio():
    df = pd.DataFrame({"temp_ext": [5, 10, 15, 20]})
    kpi = KPIEngine(price_energy=0.18)
    deg = kpi.degree_day_ratio(df)
    assert deg is not None
    assert deg < 5  # ramené à la journée (division par 24)


def test_thermal_loss():
    kpi = KPIEngine(price_energy=0.18)
    loss = kpi.thermal_loss(volume_m3=200, tau_hours=5)
    assert loss > 0
    override = kpi.thermal_loss(volume_m3=200, tau_hours=5, loss_w_per_k=120)
    assert override == 120

def test_anomaly_index():
    series = pd.Series([10, 12, 13, 50])
    kpi = KPIEngine(price_energy=0.18)
    z = kpi.anomaly_index(series)
    assert z != 0
