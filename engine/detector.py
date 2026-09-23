"""Layers 2 + 3 - hybrid Isolation Forest -> Random Forest detector.

Decision rule (score = -IsolationForest.score_samples(x), p = share of RF trees voting
for the predicted class yhat):

    score <  theta                      -> Benign                (not flagged, only counted)
    yhat == Benign and p >= tau         -> Benign                (cleared false alarm)
    p >= tau                            -> yhat                  (sent to the policy engine)
    p <  tau                            -> "Unknown anomaly"     (SOC alert, no auto-block)

theta defaults to the 99th percentile of benign *validation* anomaly scores; the
validation scores are stored in the model file so theta can be re-tuned (admin page /
config) without retraining.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from engine.schema import BENIGN, CATEGORY_SEVERITY, UNKNOWN_ANOMALY

ROUTE_BELOW_THRESHOLD = "below_threshold"   # IF says normal -> counted only
ROUTE_CLEARED = "cleared_false_alarm"       # IF anomalous, RF confidently benign
ROUTE_CLASSIFIED = "classified_attack"      # IF anomalous, RF confident attack -> policy
ROUTE_UNKNOWN = "unknown_anomaly"           # IF anomalous, RF unsure -> alert only
FLAGGED_ROUTES = (ROUTE_CLASSIFIED, ROUTE_UNKNOWN)


def decide(score: float, theta: float, rf_class: str | None, confidence: float | None,
           tau: float) -> tuple[str, str]:
    """Pure decision rule for a single flow. Returns (route, final_label)."""
    if score < theta:
        return ROUTE_BELOW_THRESHOLD, BENIGN
    if rf_class is None or confidence is None:
        raise ValueError("flow is above theta: an RF prediction is required")
    if confidence >= tau:
        if rf_class == BENIGN:
            return ROUTE_CLEARED, BENIGN
        return ROUTE_CLASSIFIED, rf_class
    return ROUTE_UNKNOWN, UNKNOWN_ANOMALY


def decide_vectorised(scores, theta, rf_classes, confidences, tau):
    """Array version of ``decide`` (identical semantics). Returns (routes, labels)."""
    scores = np.asarray(scores, dtype=float)
    rf_classes = np.asarray(rf_classes, dtype=object)
    confidences = np.asarray(confidences, dtype=float)
    above = scores >= theta
    confident = confidences >= tau
    is_benign = rf_classes == BENIGN
    routes = np.full(len(scores), ROUTE_BELOW_THRESHOLD, dtype=object)
    labels = np.full(len(scores), BENIGN, dtype=object)
    cleared = above & confident & is_benign
    classified = above & confident & ~is_benign
    unknown = above & ~confident
    routes[cleared] = ROUTE_CLEARED
    routes[classified] = ROUTE_CLASSIFIED
    labels[classified] = rf_classes[classified]
    routes[unknown] = ROUTE_UNKNOWN
    labels[unknown] = UNKNOWN_ANOMALY
    return routes, labels


def tree_vote_share(rf, X) -> tuple[np.ndarray, np.ndarray]:
    """Hard-vote the forest: (predicted class, share of trees voting for it).

    The sub-trees of a fitted RandomForestClassifier predict *encoded* class indices,
    so their predictions index ``rf.classes_`` directly.
    """
    n = X.shape[0]
    counts = np.zeros((n, len(rf.classes_)), dtype=np.int32)
    rows = np.arange(n)
    for est in rf.estimators_:
        counts[rows, est.predict(X).astype(np.int64)] += 1
    best = counts.argmax(axis=1)
    return rf.classes_[best], counts[rows, best] / len(rf.estimators_)


def theta_from_scores(benign_val_scores, percentile: float) -> float:
    return float(np.percentile(np.asarray(benign_val_scores, dtype=float), percentile))


@dataclass
class DetectionResult:
    frame: pd.DataFrame  # per-flow: anomaly_score, rf_class, confidence, route, label, severity, repaired
    theta: float
    tau: float

    def counts(self) -> dict:
        return {
            "total": int(len(self.frame)),
            "below_threshold": int((self.frame["route"] == ROUTE_BELOW_THRESHOLD).sum()),
            "cleared_false_alarm": int((self.frame["route"] == ROUTE_CLEARED).sum()),
            "classified_attack": int((self.frame["route"] == ROUTE_CLASSIFIED).sum()),
            "unknown_anomaly": int((self.frame["route"] == ROUTE_UNKNOWN).sum()),
            "repaired_rows": int(self.frame["repaired"].sum()),
        }


class HybridDetector:
    def __init__(self, if_artifact: dict, rf_artifact: dict, tau: float = 0.70,
                 theta_percentile: float | None = None):
        if if_artifact.get("kind") != "isolation_forest" or rf_artifact.get("kind") != "random_forest":
            raise ValueError("expected an isolation_forest and a random_forest artifact")
        if list(if_artifact["feature_names"]) != list(rf_artifact["feature_names"]):
            raise ValueError("Isolation Forest and Random Forest were trained on different "
                             "feature sets - activate models from the same training run")
        self.if_artifact = if_artifact
        self.rf_artifact = rf_artifact
        self.preprocessor = if_artifact["preprocessor"]
        self.iforest = if_artifact["model"]
        self.rf = rf_artifact["model"]
        self.tau = float(tau)
        self.theta_percentile = float(theta_percentile if theta_percentile is not None
                                      else if_artifact["theta_percentile"])
        self.theta = theta_from_scores(if_artifact["benign_val_scores"], self.theta_percentile)

    @classmethod
    def from_files(cls, if_path, rf_path, **kwargs) -> "HybridDetector":
        return cls(joblib.load(Path(if_path)), joblib.load(Path(rf_path)), **kwargs)

    @property
    def feature_names(self) -> list[str]:
        return list(self.if_artifact["feature_names"])

    @property
    def data_source(self) -> str:
        return self.if_artifact.get("data_source", "unknown")

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        return -self.iforest.score_samples(X)

    def detect(self, features: pd.DataFrame, batch_size: int = 20000) -> DetectionResult:
        parts = []
        for start in range(0, len(features), batch_size):
            chunk = features.iloc[start:start + batch_size]
            X, repaired = self.preprocessor.transform(chunk, return_repaired=True)
            scores = self.anomaly_scores(X)
            rf_class = np.full(len(chunk), None, dtype=object)
            conf = np.full(len(chunk), np.nan)
            above = scores >= self.theta
            if above.any():  # RF only runs on flows that pass the anomaly gate
                rf_class[above], conf[above] = tree_vote_share(self.rf, X[above])
            routes, labels = decide_vectorised(scores, self.theta, rf_class,
                                               np.nan_to_num(conf, nan=-1.0), self.tau)
            parts.append(pd.DataFrame({
                "anomaly_score": scores, "rf_class": rf_class, "confidence": conf,
                "route": routes, "label": labels, "repaired": repaired,
            }, index=chunk.index))
        frame = pd.concat(parts) if parts else pd.DataFrame(
            columns=["anomaly_score", "rf_class", "confidence", "route", "label", "repaired"])
        frame["severity"] = [
            CATEGORY_SEVERITY.get(lbl, "Medium") if r in FLAGGED_ROUTES else None
            for r, lbl in zip(frame["route"], frame["label"])
        ]
        return DetectionResult(frame=frame, theta=self.theta, tau=self.tau)
