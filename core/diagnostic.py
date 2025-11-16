class Diagnostic:
    def __init__(self, thermal_loss, isolation_level):
        self.loss = thermal_loss
        self.iso = isolation_level

    def classify(self):
        if self.loss < 80:
            return "faible"
        if self.loss < 150:
            return "moyenne"
        return "forte"

    def severity_score(self):
        if self.loss < 80:
            return 1
        if self.loss < 150:
            return 2
        return 3

    def explanation(self):
        if self.loss < 80:
            return "Structure peu sensible aux variations extérieures."
        if self.loss < 150:
            return "Maison modérément exposée à la météo et au vent."
        return "Habitation très sensible aux variations extérieures."

    def wind_impact(self, mean_wind):
        impact = self.loss * (0.02 * mean_wind)
        return round(impact, 2)

    def recommendation(self):
        cls = self.classify()
        if cls == "faible":
            return "Ajuster uniquement les horaires du poêle selon la météo."
        if cls == "moyenne":
            return "Réduire les infiltrations d’air et surveiller les jours venteux."
        return "L'amélioration de l'isolation ou le renforcement des seuils peut réduire significativement les pertes."

    def summary(self):
        return {
            "classe": self.classify(),
            "score": self.severity_score(),
            "explication": self.explanation(),
            "recommandation": self.recommendation()
        }
