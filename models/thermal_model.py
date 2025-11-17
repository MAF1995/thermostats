from __future__ import annotations

import numpy as np
import pandas as pd


class ThermalModel:
    """1R-1C thermal model describing the indoor temperature dynamics."""

    def __init__(
        self,
        tau_hours=None,
        volume_m3=None,
        power_kw=None,
        efficiency=None,
        infiltration_factor: float = 1.0,
        capacitance_kwh: float | None = None,
        loss_w_per_k: float | None = None,
        **legacy_kwargs,
    ):
        """
        Accept both the new ``tau_hours`` keyword and legacy positional/keyword arguments.

        Additional parameters allow the RC model to integrate infiltration (VMC) and
        inertial presets coming from the structure selection.
        """

        if tau_hours is None:
            tau_hours = legacy_kwargs.pop("tau", None)

        if volume_m3 is None and legacy_kwargs:
            volume_m3 = legacy_kwargs.pop("volume_m3", None) or legacy_kwargs.pop("volume", None)
        if power_kw is None and legacy_kwargs:
            power_kw = legacy_kwargs.pop("power_kw", None) or legacy_kwargs.pop("power", None)
        if efficiency is None and legacy_kwargs:
            efficiency = legacy_kwargs.pop("efficiency", None) or legacy_kwargs.pop("eff", None)

        missing = [
            name for name, val in {
                "tau_hours": tau_hours,
                "volume_m3": volume_m3,
                "power_kw": power_kw,
                "efficiency": efficiency,
            }.items()
            if val is None
        ]
        if missing:
            raise ValueError(f"ThermalModel missing required parameters: {', '.join(missing)}")

        self.tau = tau_hours
        self.volume = volume_m3
        self.power = power_kw
        self.eff = efficiency
        self.infiltration_factor = infiltration_factor
        # Volumetric heat capacity of air ~0.34 Wh/(m³·K)
        # volume_m3 is validated above; assert to help type checkers that it's not None
        assert volume_m3 is not None
        base_C = 0.34 * float(volume_m3) / 1000  # kWh/K
        self.C = capacitance_kwh if capacitance_kwh is not None else base_C
        self.loss_w_per_k = loss_w_per_k

    @property
    def _capacitance_kwh(self):
        """Heat capacity expressed in kWh/K for internal calculations."""
        return self.C

    @property
    def _heating_rate(self):
        """Equivalent temperature gain per hour from the stove (K/h)."""
        if self._capacitance_kwh == 0:
            return 0.0
        # ensure power and efficiency are numeric (fallback to 0.0 if None)
        power = 0.0 if self.power is None else float(self.power)
        eff = 0.0 if self.eff is None else float(self.eff)
        return (power * eff) / self._capacitance_kwh

    def thermal_resistance(self):
        """Return the lumped thermal resistance (h·K/kWh)."""
        if self.C == 0:
            return np.inf
        return self.tau / self.C

    def heat_capacity(self):
        """Return the lumped heat capacity (kWh/K)."""
        return self.C

    def _tau_effective(self):
        if self.loss_w_per_k:
            g_kwh_per_hk = self.loss_w_per_k / 1000
            if self.C == 0 or g_kwh_per_hk == 0:
                return self.tau
            return max(0.5, (self.C / g_kwh_per_hk) / self.infiltration_factor)
        return self.tau / max(self.infiltration_factor, 1e-3)

    def heat_step(self, T_int, T_ext_eff, dt_hours, max_delta_per_hour: float | None = None):
        tau_eff = self._tau_effective()
        dT_env = (T_ext_eff - T_int) * (dt_hours / tau_eff)
        dT_poele = self._heating_rate * dt_hours
        delta = dT_env + dT_poele
        if max_delta_per_hour is not None and max_delta_per_hour > 0:
            cap = max_delta_per_hour * dt_hours
            if delta > cap:
                delta = cap
        return T_int + delta

    def cool_step(self, T_int, T_ext_eff, dt_hours):
        tau_eff = self._tau_effective()
        return T_int + (T_ext_eff - T_int) * (dt_hours / tau_eff)

    def simulate(
        self,
        T_int_start,
        ext_series,
        wind_series,
        hours_on,
        humidity_series=None,
        target_temp=None,
        buffer=0.5,
        return_details: bool = False,
        custom_mask=None,
        max_delta_per_hour: float | None = None,
    ):
        results = []
        stove_mask = []
        T_int = T_int_start

        for i in range(len(ext_series)):
            T_ext = ext_series[i]
            wind = wind_series[i]
            T_eff = T_ext - 0.2 * wind
            humidity = None
            if humidity_series is not None and i < len(humidity_series):
                humidity = humidity_series[i]

            if custom_mask is not None:
                stove_on = bool(custom_mask[i]) if i < len(custom_mask) else False
            else:
                stove_on = i < hours_on
            if target_temp is not None and T_int >= target_temp + buffer:
                stove_on = False
            stove_mask.append(bool(stove_on))

            if stove_on:
                T_int = self.heat_step(T_int, T_eff, 1, max_delta_per_hour=max_delta_per_hour)
                if target_temp is not None:
                    T_int = min(T_int, target_temp)
            else:
                T_int = self.cool_step(T_int, T_eff, 1)
                extra_loss = self._extra_cooling(T_int, T_ext, wind, humidity)
                T_int -= extra_loss
                if target_temp is not None:
                    T_int = min(T_int, target_temp)

            results.append(T_int)

        temps = pd.Series(results)
        if return_details:
            return pd.DataFrame({"temp": temps, "stove_on": stove_mask})
        return temps

    def _extra_cooling(self, T_int, T_ext, wind, humidity):
        delta = max(0.0, T_int - T_ext)
        if delta == 0:
            return 0.0
        wind_term = min(1.2, wind / 40.0)
        humidity_term = 0.0
        if humidity is not None:
            humidity_term = max(0.0, (humidity - 60.0) / 40.0)
        structure_term = max(0.5, min(1.8, self.infiltration_factor))
        modifier = 1 + 0.6 * wind_term + 0.4 * humidity_term
        return delta * 0.015 * modifier * structure_term

    def time_to_reach(self, T_int, T_target, T_ext_eff, max_delta_per_hour: float | None = None):
        if T_target <= T_int:
            return 0
        heating_rate = self._heating_rate
        if max_delta_per_hour is not None and max_delta_per_hour > 0:
            heating_rate = min(heating_rate, max_delta_per_hour)
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

    def time_series_until_target(
        self,
        T_int,
        T_target,
        T_ext_eff_series,
        max_delta_per_hour: float | None = None,
    ):
        temps = []
        T = T_int
        for i, T_eff in enumerate(T_ext_eff_series):
            if T >= T_target:
                break
            T = self.heat_step(T, T_eff, 1, max_delta_per_hour=max_delta_per_hour)
            temps.append(T)
        return temps

