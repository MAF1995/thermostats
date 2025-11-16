import pandas as pd
from core.kpi_engine import KPIEngine

def test_daily_cost():
    kpi = KPIEngine(price_energy=0.18)
    cost = kpi.daily_cost(hours_on=4, power_kw=8)
    assert cost == 4 * 8 * 0.18

def test_degree_day_ratio():
    df = pd.DataFrame({"temp_ext": [5, 10, 15, 20]})
    kpi = KPIEngine(price_energy=0.18)
    deg = kpi.degree_day_ratio(df)
    assert deg is not None
    assert deg > 0


def test_thermal_loss():
    kpi = KPIEngine(price_energy=0.18)
    loss = kpi.thermal_loss(volume_m3=200, tau_hours=5)
    assert loss > 0

def test_anomaly_index():
    series = pd.Series([10, 12, 13, 50])
    kpi = KPIEngine(price_energy=0.18)
    z = kpi.anomaly_index(series)
    assert z != 0
