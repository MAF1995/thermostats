from __future__ import annotations

import pandas as pd

PELLET_ENERGY_KWH_PER_KG = 4.6
KG_PER_BAG = 15


class PelletEngine:
    """Calculs de consommation granulés et des points de recharge."""

    def __init__(self, power_kw: float, efficiency: float, pellet_price_bag: float):
        self.power_kw = power_kw
        self.efficiency = efficiency
        self.pellet_price_bag = pellet_price_bag

    def hourly_bag_rate(
        self,
        target_temp: float,
        hours_on: int,
        desired_duration_hours: float | None = None,
    ) -> float:
        bounds = self.bag_duration_bounds()
        nominal_duration = desired_duration_hours or bounds["nominal_hours"]
        base_rate = 1.0 / max(1.0, nominal_duration)

        temp_delta = target_temp - 21.0
        if temp_delta >= 0:
            temp_factor = 1 + temp_delta * 0.035
        else:
            temp_factor = 1 + temp_delta * 0.02
        temp_factor = max(0.7, min(1.4, temp_factor))

        heating_hours = max(1, hours_on)
        workload_factor = 1 + (heating_hours / 24.0) * 0.02

        rate = base_rate * temp_factor * workload_factor
        min_rate = 1 / bounds["max_hours"]
        max_rate = 1 / bounds["min_hours"]
        return min(max(rate, min_rate), max_rate)

    def compute_pellet_usage(
        self,
        hours: int,
        target_temp: float,
        hours_on: int,
        desired_duration_hours: float,
        active_mask=None,
    ) -> pd.DataFrame:
        if active_mask is None:
            active_mask = [True] * hours

        rate_bag_h = self.hourly_bag_rate(target_temp, hours_on, desired_duration_hours)
        data = []
        bags_consumed = 0.0
        previous_whole = 0
        segment = 0
        for h in range(hours):
            burning = active_mask[h] if h < len(active_mask) else False
            used = rate_bag_h if burning else 0.0
            new_total = bags_consumed + used
            whole_before = int(bags_consumed)
            whole_after = int(new_total)
            bags_consumed = new_total
            if whole_after > segment:
                segment = whole_after
            recharge_flag = burning and (whole_after > whole_before)
            data.append(
                {
                    "heure": h,
                    "bags_used": used,
                    "bags_cum": bags_consumed,
                    "recharge": recharge_flag,
                    "segment": segment,
                }
            )

        df = pd.DataFrame(data)
        df["kg_cum"] = df["bags_cum"] * KG_PER_BAG
        df["kwh_cum"] = df["kg_cum"] * PELLET_ENERGY_KWH_PER_KG
        df["cost_cum"] = df["bags_cum"] * self.pellet_price_bag
        df["recharge_points"] = df["bags_cum"].apply(lambda x: int(x))
        return df

    def bag_duration_bounds(self):
        return {"min_hours": 12, "max_hours": 18, "nominal_hours": 14}

