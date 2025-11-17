class Diagnostic:
    def __init__(self, thermal_loss, config, meteo=None):
        self.loss = thermal_loss
        self.cfg = config
        self.meteo = meteo or {}

    def classify(self):
        if self.loss < 80:
            return "faible"
        if self.loss < 150:
            return "moyenne"
        return "forte"

    def severity_score(self):
        base = 1
        if self.loss >= 80:
            base = 2 if self.loss < 150 else 3
        wind = self.meteo.get("wind_mean", 0)
        humidity = self.meteo.get("humidity_mean", 50)
        weather_bonus = 0
        if wind > 20:
            weather_bonus += 0.5
        if humidity > 75:
            weather_bonus += 0.5
        return min(3, base + weather_bonus)

    def explanation(self):
        wind = self.meteo.get("wind_mean")
        humidity = self.meteo.get("humidity_mean")
        parts = []
        if self.loss < 80:
            parts.append("Structure peu sensible aux variations extérieures.")
        elif self.loss < 150:
            parts.append("Maison modérément exposée aux vents.")
        else:
            parts.append("Habitation très sensible aux pertes thermiques.")
        if wind is not None:
            parts.append(f"Vent moyen {wind:.1f} km/h amplifiant les infiltrations.")
        if humidity is not None and humidity > 70:
            parts.append("Humidité élevée favorisant le refroidissement des parois.")
        return " ".join(parts)

    def construction_profile(self):
        return (
            f"Structure: {self.cfg.structure} · Isolation: {self.cfg.isolation} · "
            f"VMC: {self.cfg.vmc} · Vitrage: {self.cfg.glazing}"
        )

    def recommendation(self):
        cls = self.classify()
        if cls == "faible":
            return "Optimiser les horaires du poêle en suivant la météo locale."
        if cls == "moyenne":
            return "Calfeutrer les points sensibles (menuiseries, VMC) les jours venteux et humides."
        return "Renforcer l'isolation, moderniser le vitrage ou passer en VMC double flux pour réduire les pertes."

    def summary(self):
        return {
            "classe": self.classify(),
            "score": self.severity_score(),
            "explication": self.explanation(),
            "recommandation": self.recommendation(),
            "construction": self.construction_profile(),
        }
