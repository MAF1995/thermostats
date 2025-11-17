from dataclasses import dataclass
from typing import Dict, List

# Combinaisons structure/isolant générées automatiquement
STRUCTURE_FAMILIES = {
    "maçonnerie": {"capacitance_kwh_m3": 0.00034, "loss_w_k": 0.9},
    "ossature bois": {"capacitance_kwh_m3": 0.00030, "loss_w_k": 0.75},
    "béton": {"capacitance_kwh_m3": 0.00045, "loss_w_k": 0.65},
    "terre crue": {"capacitance_kwh_m3": 0.00050, "loss_w_k": 0.55},
}

INSULATION_TYPES = {
    "aucun": {"cap_factor": 1.0, "loss_factor": 1.2},
    "laine minérale": {"cap_factor": 1.05, "loss_factor": 0.8},
    "ouate de cellulose": {"cap_factor": 1.08, "loss_factor": 0.78},
    "polystyrène": {"cap_factor": 1.02, "loss_factor": 0.82},
    "paille": {"cap_factor": 1.10, "loss_factor": 0.7},
    "liège": {"cap_factor": 1.06, "loss_factor": 0.76},
}

SPECIAL_STRUCTURES = {
    "Maison semi-enterrée": {"capacitance_kwh_m3": 0.00050, "loss_w_k": 0.55},
    "Serre en verre": {"capacitance_kwh_m3": 0.00028, "loss_w_k": 1.20},
    "Maison paille": {"capacitance_kwh_m3": 0.00042, "loss_w_k": 0.50},
    "Maison hobbit": {"capacitance_kwh_m3": 0.00048, "loss_w_k": 0.52},
    "Igloo": {"capacitance_kwh_m3": 0.00032, "loss_w_k": 0.72},
    "Bunker béton armé": {"capacitance_kwh_m3": 0.00060, "loss_w_k": 0.40},
}

GLAZING_TYPES = {
    "simple vitrage": 1.15,
    "double vitrage": 1.0,
    "triple vitrage": 0.9,
}

VMC_OPTIONS = {
    "aucune": 1.20,  # plus d'infiltration
    "simple flux": 1.10,
    "hygro A": 1.05,
    "hygro B": 1.00,
    "double flux": 0.90,
    "double flux thermodynamique": 0.85,
}

INERTIA_LABELS = {
    "faible": 2,
    "moyenne": 5,
    "forte": 10,
}


def _generate_presets() -> Dict[str, Dict[str, float]]:
    presets: Dict[str, Dict[str, float]] = {}
    for fam, base in STRUCTURE_FAMILIES.items():
        for ins, factors in INSULATION_TYPES.items():
            label = f"{fam} + {ins}"
            presets[label] = {
                "capacitance_kwh_m3": base["capacitance_kwh_m3"] * factors["cap_factor"],
                "loss_w_k": base["loss_w_k"] * factors["loss_factor"],
            }
    presets.update(SPECIAL_STRUCTURES)
    return presets


STRUCTURE_PRESETS: Dict[str, Dict[str, float]] = _generate_presets()


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
    glazing: str

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
        glazing_factor = GLAZING_TYPES.get(self.glazing, 1.0)
        return base * vmc_factor * glazing_factor

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
        if self.glazing not in GLAZING_TYPES:
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

    @staticmethod
    def glazing_choices() -> List[str]:
        return list(GLAZING_TYPES.keys())
