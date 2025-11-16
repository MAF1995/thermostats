from dataclasses import dataclass

INERTIA_MAP = {
    "faible": 2,
    "moyenne": 5,
    "forte": 10
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

    def tau_hours(self):
        return INERTIA_MAP.get(self.inertia_level, 5)

    def validate(self):
        if self.volume_m3 <= 0:
            return False
        if self.power_kw <= 0:
            return False
        if not (0 < self.efficiency <= 1):
            return False
        if self.temp_target <= self.temp_current:
            return False
        return True
