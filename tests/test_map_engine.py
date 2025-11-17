import pandas as pd
from core.map_engine import MapEngine


def test_calc_timelapse_frames_handles_named_layers():
    engine = MapEngine()
    df = pd.DataFrame({
        "city": ["Test"],
        "lat": [0.0],
        "lon": [0.0],
        "temp_0": [10],
        "temp_1": [11],
        "wind_0": [5],
        "wind_1": [6],
        "region": ["R1"],
        "department": ["D1"],
    })
    timeline = ["t0", "t1"]
    layers = {
        "Température": {"scale": "RdBu_r", "unit": "°C", "col": "temp"},
        "Vent": {"scale": "PuBu", "unit": "km/h", "col": "wind"},
    }

    aggregates = {"Régions": {"df": df.copy(), "label": "region"}}

    frames = engine.calc_timelapse_frames(df, timeline, layers, aggregates=aggregates)

    assert len(frames) == 2
    assert all(len(frame.data) == 4 for frame in frames)  # 2 couches + 2 agrégats


def test_offline_communes_dense():
    engine = MapEngine()
    df = engine._offline_communes()
    assert {"city", "lat", "lon", "pop", "region", "department"}.issubset(df.columns)
    assert len(df) > 30000
