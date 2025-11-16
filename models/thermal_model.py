import numpy as np
import pandas as pd


class ThermalModel:
    """1R-1C thermal model describing the indoor temperature dynamics."""

    def __init__(self, tau_hours, volume_m3, power_kw, efficiency):
        self.tau = tau_hours
        self.volume = volume_m3
        self.power = power_kw
        self.eff = efficiency
        # Volumetric heat capacity of air ~0.34 Wh/(m³·K) ≈ 0.00034 kWh/(m³·K)
        self.C = 0.34 * volume_m3 / 1000  # kWh/K

    @property
    def _capacitance_kwh(self):
        """Heat capacity expressed in kWh/K for internal calculations."""
        return self.C

    @property
    def _heating_rate(self):
        """Equivalent temperature gain per hour from the stove (K/h)."""
        if self._capacitance_kwh == 0:
            return 0.0
        return (self.power * self.eff) / self._capacitance_kwh

    def thermal_resistance(self):
        """Return the lumped thermal resistance (h·K/kWh)."""
        if self.C == 0:
            return np.inf
        return self.tau / self.C

    def heat_capacity(self):
        """Return the lumped heat capacity (kWh/K)."""
        return self.C

    def heat_step(self, T_int, T_ext_eff, dt_hours):
        # Forward Euler discretisation of dT/dt = (T_ext - T_int)/tau + P/C
        dT_env = (T_ext_eff - T_int) * (dt_hours / self.tau)
        dT_poele = self._heating_rate * dt_hours
        return T_int + dT_env + dT_poele

    def cool_step(self, T_int, T_ext_eff, dt_hours):
        return T_int + (T_ext_eff - T_int) * (dt_hours / self.tau)

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
        if T_target <= T_int:
            return 0
        heating_rate = self._heating_rate
        if heating_rate <= 0:
            return None

        steady_state = T_ext_eff + self.tau * heating_rate
        A = T_int - steady_state
        B = T_target - steady_state

        if A == 0:
            return None

        ratio = B / A
        if ratio <= 0 or ratio >= 1:
            return None

        dt = -self.tau * np.log(ratio)
        return dt
    
    def time_series_until_target(self, T_int, T_target, T_ext_eff_series):
        temps = []
        T = T_int
        for i, T_eff in enumerate(T_ext_eff_series):
            if T >= T_target:
                break
            T = self.heat_step(T, T_eff, 1)
            temps.append(T)
        return temps

