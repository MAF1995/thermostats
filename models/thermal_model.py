import numpy as np
import pandas as pd

class ThermalModel:
    def __init__(self, tau_hours, volume_m3, power_kw, efficiency):
        self.tau = tau_hours
        self.volume = volume_m3
        self.power = power_kw
        self.eff = efficiency
        self.C = 0.34 * volume_m3

    def heat_step(self, T_int, T_ext_eff, dt_hours):
        num = (self.power * 1000 * self.eff) * dt_hours
        dT = num / (self.C * 1000)
        T_target = T_ext_eff + (T_int - T_ext_eff) * np.exp(-dt_hours / self.tau)
        return T_target + dT

    def cool_step(self, T_int, T_ext_eff, dt_hours):
        return T_ext_eff + (T_int - T_ext_eff) * np.exp(-dt_hours / self.tau)

    def simulate(self, T_int_start, ext_series, wind_series, hours_on):
        results = []
        T_int = T_int_start

        for i in range(len(ext_series)):
            T_ext = ext_series[i]
            wind = wind_series[i]
            T_eff = T_ext - 0.2 * wind

            if i < hours_on:
                T_int = self.heat_step(T_int, T_eff, 1)
            else:
                T_int = self.cool_step(T_int, T_eff, 1)

            results.append(T_int)

        return pd.Series(results)

    def time_to_reach(self, T_int, T_target, T_ext_eff):
        num = (T_target - T_int) * self.C
        den = (self.power * self.eff)
        if den <= 0:
            return None
        x = 1 - (num / den)
        if x >= 1:
            return 0
        if x <= 0:
            return None
        dt = -self.tau * np.log(x)
        return dt
