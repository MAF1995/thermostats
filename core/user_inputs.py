from dataclasses import dataclass
from typing import Dict, List

# Liste experte d'assemblages muraux / structures avec inertie (C) et pertes (G)
STRUCTURE_PRESETS: Dict[str, Dict[str, float]] = {
    "BA11 + brique": {"capacitance_kwh_m3": 0.00036, "loss_w_k": 0.82},
    "BA13 + brique": {"capacitance_kwh_m3": 0.00038, "loss_w_k": 0.78},
    "Bois + laine minérale": {"capacitance_kwh_m3": 0.00030, "loss_w_k": 0.68},
    "Parpaing + BA13": {"capacitance_kwh_m3": 0.00035, "loss_w_k": 0.90},
    "Béton + isolant 20 cm": {"capacitance_kwh_m3": 0.00045, "loss_w_k": 0.60},
    "Maison semi-enterrée": {"capacitance_kwh_m3": 0.00050, "loss_w_k": 0.55},
    "Serre en verre": {"capacitance_kwh_m3": 0.00028, "loss_w_k": 1.20},
    "Maison paille": {"capacitance_kwh_m3": 0.00042, "loss_w_k": 0.50},
    "Maison hobbit": {"capacitance_kwh_m3": 0.00048, "loss_w_k": 0.52},
    "Igloo": {"capacitance_kwh_m3": 0.00032, "loss_w_k": 0.72},
    "Bunker béton armé": {"capacitance_kwh_m3": 0.00060, "loss_w_k": 0.40},
}

VMC_OPTIONS = {
    "aucune": 1.20,  # plus d'infiltration
    "simple flux": 1.10,
    "hygro A": 1.05,
    "hygro B": 1.00,
    "double flux": 0.90,
}

INERTIA_LABELS = {
    "faible": 2,
    "moyenne": 5,
    "forte": 10,
}


@dataclass
class UserConfig:
    volume_m3: float
    inertia_level: str
    power_kw: float
    efficiency: float
    temp_current: float
    temp_target: float
    latitude: float
    longitude: float
    isolation: str
    structure: str
    vmc: str
    pellet_price_bag: float

    def tau_hours(self) -> float:
        structure = STRUCTURE_PRESETS.get(self.structure)
        if structure:
            capacitance = structure["capacitance_kwh_m3"] * self.volume_m3
            g_w_k = structure["loss_w_k"]
            tau_structure = (capacitance * 1000) / max(g_w_k, 1e-3)
            return max(1.5, tau_structure / 3600)  # en heures
        return INERTIA_LABELS.get(self.inertia_level, 5)

    def capacitance_kwh(self) -> float:
        structure = STRUCTURE_PRESETS.get(self.structure)
        if structure:
            return structure["capacitance_kwh_m3"] * self.volume_m3
        return 0.34 * self.volume_m3 / 1000

    def loss_w_per_k(self) -> float:
        structure = STRUCTURE_PRESETS.get(self.structure)
        base = structure["loss_w_k"] if structure else 0.9
        vmc_factor = VMC_OPTIONS.get(self.vmc, 1.0)
        return base * vmc_factor

    def validate(self) -> bool:
        if self.volume_m3 <= 0:
            return False
        if self.power_kw <= 0:
            return False
        if not (0 < self.efficiency <= 1):
            return False
        if self.temp_target <= self.temp_current:
            return False
        if self.structure not in STRUCTURE_PRESETS:
            return False
        if self.vmc not in VMC_OPTIONS:
            return False
        return True

    @staticmethod
    def structures() -> List[str]:
        return list(STRUCTURE_PRESETS.keys())

    @staticmethod
    def vmc_choices() -> List[str]:
        return list(VMC_OPTIONS.keys())
