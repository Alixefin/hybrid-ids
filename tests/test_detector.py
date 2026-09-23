"""Detection routing logic (the hybrid decision rule) and the detector wrapper."""
import joblib
import numpy as np
import pytest

from engine.detector import (
    ROUTE_BELOW_THRESHOLD, ROUTE_CLASSIFIED, ROUTE_CLEARED, ROUTE_UNKNOWN,
    HybridDetector, decide, decide_vectorised, theta_from_scores, tree_vote_share,
)
from engine.schema import BENIGN, CAT_DOS, CAT_PORTSCAN, UNKNOWN_ANOMALY

THETA, TAU = 0.60, 0.70


@pytest.mark.parametrize("score,rf_class,p,route,label", [
    (0.40, CAT_DOS, 0.99, ROUTE_BELOW_THRESHOLD, BENIGN),     # below theta: RF output irrelevant
    (0.40, None, None, ROUTE_BELOW_THRESHOLD, BENIGN),        # RF not even needed
    (0.65, BENIGN, 0.95, ROUTE_CLEARED, BENIGN),              # anomalous but confidently benign
    (0.65, BENIGN, 0.70, ROUTE_CLEARED, BENIGN),              # p == tau counts as confident
    (0.65, CAT_DOS, 0.91, ROUTE_CLASSIFIED, CAT_DOS),         # confident attack -> policy engine
    (0.60, CAT_PORTSCAN, 0.70, ROUTE_CLASSIFIED, CAT_PORTSCAN),  # score == theta is anomalous
    (0.65, CAT_DOS, 0.69, ROUTE_UNKNOWN, UNKNOWN_ANOMALY),    # unsure attack -> unknown
    (0.65, BENIGN, 0.55, ROUTE_UNKNOWN, UNKNOWN_ANOMALY),     # unsure benign -> unknown too
])
def test_decision_rule(score, rf_class, p, route, label):
    assert decide(score, THETA, rf_class, p, TAU) == (route, label)


def test_above_theta_requires_rf_output():
    with pytest.raises(ValueError):
        decide(0.9, THETA, None, None, TAU)


def test_vectorised_rule_matches_scalar_rule(rng):
    n = 2000
    scores = rng.uniform(0.3, 0.9, n)
    classes = rng.choice([BENIGN, CAT_DOS, CAT_PORTSCAN], n).astype(object)
    confs = rng.choice([0.5, 0.69, 0.7, 0.71, 0.9, 1.0], n)
    routes, labels = decide_vectorised(scores, THETA, classes, confs, TAU)
    expected = [decide(s, THETA, c, p, TAU) for s, c, p in zip(scores, classes, confs)]
    assert list(zip(routes, labels)) == expected


def test_tree_vote_share_is_hard_vote(tiny_run):
    fp = tiny_run["fp"]
    X = fp.preprocessor.transform(tiny_run["test"].iloc[:300])
    classes, share = tree_vote_share(fp.rf, X)
    # recount the votes by hand
    votes = np.stack([fp.rf.classes_[est.predict(X).astype(int)] for est in fp.rf.estimators_])
    for i in range(0, 300, 37):
        vals, counts = np.unique(votes[:, i], return_counts=True)
        assert counts.max() / len(fp.rf.estimators_) == pytest.approx(share[i])
        assert classes[i] in vals[counts == counts.max()]
    assert ((share > 0) & (share <= 1)).all()


def test_theta_percentile():
    scores = np.arange(1, 101, dtype=float)
    assert theta_from_scores(scores, 99) == pytest.approx(99.01)
    assert theta_from_scores(scores, 50) == pytest.approx(50.5)


def _artifacts(tiny_run):
    d, m = tiny_run["dir"], tiny_run["manifest"]
    return joblib.load(d / m["isolation_forest"]["file"]), joblib.load(d / m["random_forest"]["file"])


def test_detector_routes_consistently_and_only_runs_rf_above_theta(tiny_run, monkeypatch):
    if_art, rf_art = _artifacts(tiny_run)
    det = HybridDetector(if_art, rf_art, tau=TAU)
    test = tiny_run["test"]
    seen = {}
    import engine.detector as mod
    real = mod.tree_vote_share

    def spy(rf, X):
        seen["rows"] = seen.get("rows", 0) + X.shape[0]
        return real(rf, X)
    monkeypatch.setattr(mod, "tree_vote_share", spy)

    res = det.detect(test, batch_size=500)
    f = res.frame
    above = f["anomaly_score"] >= det.theta
    assert seen.get("rows", 0) == int(above.sum())                   # RF only on gated flows
    assert (f.loc[~above, "route"] == ROUTE_BELOW_THRESHOLD).all()
    assert f.loc[~above, "rf_class"].isna().all()
    for _, r in f[above].iterrows():
        assert (r["route"], r["label"]) == decide(r["anomaly_score"], det.theta, r["rf_class"], r["confidence"], TAU)
    c = res.counts()
    assert c["total"] == len(test) == sum(c[k] for k in (
        "below_threshold", "cleared_false_alarm", "classified_attack", "unknown_anomaly"))
    assert f.loc[f["route"] == ROUTE_UNKNOWN, "label"].eq(UNKNOWN_ANOMALY).all()
    # severity only for flagged flows
    assert f.loc[f["route"].isin([ROUTE_BELOW_THRESHOLD, ROUTE_CLEARED]), "severity"].isna().all()


def test_detector_catches_attacks_on_synthetic_data(tiny_run):
    if_art, rf_art = _artifacts(tiny_run)
    det = HybridDetector(if_art, rf_art, tau=TAU)
    test = tiny_run["test"]
    f = det.detect(test).frame
    scan = (test["Category"] == CAT_PORTSCAN).to_numpy()
    assert (f["route"].to_numpy()[scan] != ROUTE_BELOW_THRESHOLD).mean() > 0.8
    benign = (test["Category"] == BENIGN).to_numpy()
    assert (f["route"].to_numpy()[benign] == ROUTE_CLASSIFIED).mean() < 0.02


def test_theta_percentile_override_changes_gate(tiny_run):
    if_art, rf_art = _artifacts(tiny_run)
    strict = HybridDetector(if_art, rf_art, theta_percentile=99.5)
    loose = HybridDetector(if_art, rf_art, theta_percentile=90)
    assert loose.theta < strict.theta
    assert strict.theta_percentile == 99.5


def test_detector_rejects_mismatched_models(tiny_run):
    if_art, rf_art = _artifacts(tiny_run)
    bad = dict(rf_art, feature_names=list(rf_art["feature_names"])[::-1])
    with pytest.raises(ValueError, match="different"):
        HybridDetector(if_art, bad)
    with pytest.raises(ValueError):
        HybridDetector(rf_art, if_art)


def test_detector_handles_non_finite_input(tiny_run):
    if_art, rf_art = _artifacts(tiny_run)
    det = HybridDetector(if_art, rf_art)
    rows = tiny_run["test"].iloc[:5].copy()
    col = det.feature_names[0]
    rows.loc[rows.index[0], col] = np.inf
    rows.loc[rows.index[1], col] = np.nan
    f = det.detect(rows).frame
    assert f["repaired"].tolist()[:2] == [True, True]
    assert np.isfinite(f["anomaly_score"]).all()
