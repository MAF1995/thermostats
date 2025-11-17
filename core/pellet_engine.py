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
        surcharge = max(0, target_temp - 20) * 0.07
        base_rate_kg_h = (self.power_kw / (PELLET_ENERGY_KWH_PER_KG * max(self.efficiency, 1e-6)))
        kg_with_temp = base_rate_kg_h * (1 + surcharge)

        heating_hours = max(1, hours_on)
        heat_factor = 1 + (max(0, 6 - heating_hours) * 0.05) - (max(0, heating_hours - 10) * 0.02)
        heat_factor = max(0.75, min(1.25, heat_factor))

        target_duration = desired_duration_hours or bounds["nominal_hours"]
        duration_factor = bounds["nominal_hours"] / max(1.0, target_duration)
        min_factor = bounds["min_hours"] / bounds["max_hours"]
        max_factor = bounds["max_hours"] / bounds["min_hours"]
        duration_factor = max(min_factor, min(max_factor, duration_factor))

        kg_adjusted = kg_with_temp * heat_factor * duration_factor
        rate = kg_adjusted / KG_PER_BAG
        rate = min(max(rate, 1 / bounds["max_hours"]), 1 / bounds["min_hours"])
        return rate

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
        segment = 0
        for h in range(hours):
            burning = active_mask[h] if h < len(active_mask) else False
            used = rate_bag_h if burning else 0.0
            bags_consumed += used
            if bags_consumed >= (segment + 1):
                segment += 1
            data.append(
                {
                    "heure": h,
                    "bags_used": used,
                    "bags_cum": bags_consumed,
                    "recharge": bags_consumed >= 1 and (bags_consumed - used) < 1,
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
        return {"min_hours": 14, "max_hours": 22, "nominal_hours": 18}

