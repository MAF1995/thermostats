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

    def recommendation(self):
        cls = self.classify()

        if cls == "faible":
            return "L'inertie thermique est suffisante pour une stabilité correcte. Un ajustement fin des horaires d’allumage du poêle est généralement suffisant."
        
        if cls == "moyenne":
            return "Les pertes thermiques sont modérées. Une surveillance accrue les jours venteux peut améliorer le confort et la consommation."
        
        return "Les pertes thermiques sont élevées. L'habitation est sensible aux variations climatiques. Une optimisation du chauffage ou une amélioration de l'isolation peut réduire les coûts."

    def summary(self):
        return {
            "classe": self.classify(),
            "score": self.severity_score(),
            "recommandation": self.recommendation()
        }
