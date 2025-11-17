import numpy as np
import pandas as pd


class KPIEngine:
    def __init__(self, price_energy):
        self.price = price_energy

    def daily_cost(self, pellet_df: pd.DataFrame, hours_on: int, standby_watts: float = 60):
        pellet_cost = 0.0 if pellet_df.empty else float(pellet_df["cost_cum"].iloc[-1])
        electric_cost = self.stove_electric_cost(hours_on, standby_watts=standby_watts)
        return pellet_cost + electric_cost

    def degree_day_ratio(self, df_temp, T_base=18):
        df = df_temp.copy()
        df["deg"] = np.maximum(0, T_base - df["temp_ext"])
        deg_sum = df["deg"].sum()
        if deg_sum == 0:
            return None
        return deg_sum / 24  # exprimé en degrés-jours

    def thermal_loss(self, volume_m3, tau_hours, loss_w_per_k: float | None = None):
        if loss_w_per_k is not None:
            return loss_w_per_k
        C = 0.34 * volume_m3
        return C / tau_hours

    def anomaly_index(self, consumption_series):
        mean = consumption_series.mean()
        std = consumption_series.std()
        if std == 0:
            return 0
        z = (consumption_series.iloc[-1] - mean) / std
        return z

    def consumption_trend(self, consumption_series):
        if len(consumption_series) < 2:
            return None
        x = np.arange(len(consumption_series))
        y = consumption_series.values
        coeffs = np.polyfit(x, y, 1)
        return coeffs[0]

    def pellet_cost(self, pellet_df: pd.DataFrame):
        if pellet_df.empty:
            return 0.0
        return float(pellet_df["cost_cum"].iloc[-1])

    def stove_electric_cost(self, hours_on: int, standby_watts: float = 60):
        kwh = (standby_watts / 1000) * hours_on
        return kwh * self.price
