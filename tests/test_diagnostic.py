from core.diagnostic import Diagnostic

def test_classification():
    d1 = Diagnostic(thermal_loss=50, isolation_level="moyenne")
    d2 = Diagnostic(thermal_loss=120, isolation_level="faible")
    d3 = Diagnostic(thermal_loss=200, isolation_level="forte")
    assert d1.classify() == "faible"
    assert d2.classify() == "moyenne"
    assert d3.classify() == "forte"

def test_recommendation():
    d = Diagnostic(thermal_loss=170, isolation_level="forte")
    rec = d.recommendation()
    assert isinstance(rec, str)

def test_summary():
    d = Diagnostic(thermal_loss=90, isolation_level="faible")
    res = d.summary()
    assert "classe" in res
    assert "score" in res
    assert "recommandation" in res
