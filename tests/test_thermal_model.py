import numpy as np
from models.thermal_model import ThermalModel

def test_heat_step():
    model = ThermalModel(tau_hours=5, volume_m3=200, power_kw=8, efficiency=0.85)
    T = model.heat_step(18, 5, 1)
    assert T > 18

def test_cool_step():
    model = ThermalModel(tau_hours=5, volume_m3=200, power_kw=8, efficiency=0.85)
    T = model.cool_step(20, 10, 1)
    assert T < 20

def test_time_to_reach_basic():
    model = ThermalModel(tau_hours=5, volume_m3=200, power_kw=8, efficiency=0.85)
    dt = model.time_to_reach(17, 20, 5)
    assert dt is not None
    assert dt > 0

def test_time_to_reach_impossible():
    model = ThermalModel(tau_hours=5, volume_m3=200, power_kw=0, efficiency=0.85)
    dt = model.time_to_reach(17, 20, 5)
    assert dt is None
