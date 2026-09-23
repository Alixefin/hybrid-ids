"""Shared training / scoring / metrics code used by train_models.py and evaluate.py."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from engine.detector import (
    FLAGGED_ROUTES, ROUTE_BELOW_THRESHOLD, ROUTE_CLASSIFIED, ROUTE_CLEARED, ROUTE_UNKNOWN,
    decide_vectorised, theta_from_scores, tree_vote_share,
)
from engine.preprocess import FlowPreprocessor, clean_rows
from engine.schema import BENIGN, FEATURE_COLUMNS, UNKNOWN_ANOMALY

DEFAULT_PARAMS = {
    "preprocess": {"n_features": 30, "corr_threshold": 0.95, "clip_quantile": 0.999,
                   "near_constant_share": 0.9995, "rf_estimators": 100, "max_fit_rows": 300_000},
    # max_samples=2048 instead of sklearn's 256: benign traffic is very heterogeneous and
    # the larger sub-sample lets IF isolate attacks that are only unusual in combination
    # (see the stage-1 notes in README).
    "isolation_forest": {"n_estimators": 200, "max_samples": 2048, "max_features": 1.0,
                         "bootstrap": False},
    "random_forest": {"n_estimators": 100, "max_depth": 30, "min_samples_leaf": 1,
                      "max_features": "sqrt", "class_weight": "balanced"},
    "theta_percentile": 99.0,
    "tau": 0.70,
    "max_benign_val_scores": 20_000,
}


def stratify_key(df: pd.DataFrame) -> pd.Series:
    """Stratify on the fine-grained label when every label has >= 10 rows, else category."""
    counts = df["Label"].value_counts()
    return df["Label"] if counts.min() >= 10 else df["Category"]


def split_random(df, sizes=(0.70, 0.15, 0.15), seed=42):
    train, rest = train_test_split(df, train_size=sizes[0], stratify=stratify_key(df), random_state=seed)
    val_share = sizes[1] / (sizes[1] + sizes[2])
    val, test = train_test_split(rest, train_size=val_share, stratify=stratify_key(rest), random_state=seed)
    return train, val, test


@dataclass
class FittedPipeline:
    preprocessor: FlowPreprocessor
    iforest: IsolationForest
    rf: RandomForestClassifier
    benign_val_scores: np.ndarray
    theta_percentile: float
    params: dict
    timings: dict = field(default_factory=dict)
    train_counts: dict = field(default_factory=dict)

    @property
    def theta(self) -> float:
        return theta_from_scores(self.benign_val_scores, self.theta_percentile)


def fit_pipeline(train: pd.DataFrame, val: pd.DataFrame, params=None, seed=42,
                 verbose=True) -> FittedPipeline:
    p = {**DEFAULT_PARAMS, **(params or {})}
    log = print if verbose else (lambda *a, **k: None)
    t = {}

    t0 = time.time()
    pre = FlowPreprocessor(random_state=seed, **p["preprocess"]).fit(train, train["Category"])
    t["preprocess_s"] = round(time.time() - t0, 2)
    log(f"    pre-processor: {pre.n_features_in_} -> {len(pre.selected_features_)} features "
        f"({len(pre.dropped_constant_)} near-constant, {len(pre.dropped_correlated_)} correlated dropped)")

    X_train = pre.transform(train)
    benign_mask = (train["Category"] == BENIGN).to_numpy()

    t0 = time.time()
    iso = IsolationForest(random_state=seed, n_jobs=-1, **p["isolation_forest"])
    iso.fit(X_train[benign_mask])  # benign flows only
    t["isolation_forest_s"] = round(time.time() - t0, 2)

    val_benign = val[val["Category"] == BENIGN]
    scores = -iso.score_samples(pre.transform(val_benign))
    rng = np.random.default_rng(seed)
    if len(scores) > p["max_benign_val_scores"]:
        scores = rng.choice(scores, p["max_benign_val_scores"], replace=False)
    scores = np.sort(scores).astype(np.float32)

    t0 = time.time()
    rf = RandomForestClassifier(random_state=seed, n_jobs=-1, **p["random_forest"])
    rf.fit(X_train, train["Category"].to_numpy())
    t["random_forest_s"] = round(time.time() - t0, 2)

    fp = FittedPipeline(pre, iso, rf, scores, p["theta_percentile"], p, t,
                        {k: int(v) for k, v in train["Category"].value_counts().items()})
    log(f"    IF on {int(benign_mask.sum()):,} benign flows; theta(p{fp.theta_percentile:g}) = "
        f"{fp.theta:.4f}; RF on {len(train):,} flows / {len(rf.classes_)} classes")
    return fp


@dataclass
class Scored:
    y_true: np.ndarray       # category
    scores: np.ndarray       # IF anomaly score
    rf_class: np.ndarray     # RF hard-vote class for every flow (RF-alone baseline)
    confidence: np.ndarray   # RF tree-vote share

    def hybrid(self, theta: float, tau: float):
        return decide_vectorised(self.scores, theta, self.rf_class, self.confidence, tau)


def score(fp: FittedPipeline, test: pd.DataFrame) -> Scored:
    X = fp.preprocessor.transform(test)
    rf_class, conf = tree_vote_share(fp.rf, X)
    return Scored(test["Category"].to_numpy(), -fp.iforest.score_samples(X), rf_class, conf)


# ----------------------------------------------------------------------------- metrics
def binary_metrics(true_attack, pred_attack) -> dict:
    t, p = np.asarray(true_attack, bool), np.asarray(pred_attack, bool)
    tp, fp = int((t & p).sum()), int((~t & p).sum())
    fn, tn = int((t & ~p).sum()), int((~t & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "fpr": round(fpr, 6), "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def multiclass_metrics(y_true, y_pred) -> dict:
    """Per-class and macro precision / recall / F1 / FPR (one-vs-rest).

    Macro averages are over the classes present in y_true. Predictions of classes not in
    y_true (e.g. "Unknown anomaly") count as errors for the true class.
    """
    y_true, y_pred = np.asarray(y_true, object), np.asarray(y_pred, object)
    true_classes = sorted(set(y_true))
    all_labels = true_classes + sorted(set(y_pred) - set(true_classes))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=true_classes, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    n = cm.sum()
    per_class = {}
    for i, c in enumerate(true_classes):
        fp = cm[:, i].sum() - cm[i, i]
        tn = n - cm[i, :].sum() - fp
        per_class[c] = {"precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4),
                        "f1": round(float(f1[i]), 4),
                        "fpr": round(float(fp / (fp + tn)) if fp + tn else 0.0, 6),
                        "support": int(sup[i])}
    macro = {k: round(float(np.mean([v[k] for v in per_class.values()])), 4)
             for k in ("precision", "recall", "f1", "fpr")}
    return {"per_class": per_class, "macro": macro,
            "accuracy": round(float((y_true == y_pred).mean()), 4),
            "confusion_matrix": {"labels": all_labels, "matrix": cm.tolist()}}


def system_metrics(s: Scored, theta: float, tau: float) -> dict:
    """Metrics for IF alone, RF alone and the hybrid pipeline on one scored test set."""
    attack = s.y_true != BENIGN
    if_flag = s.scores >= theta
    if_per_cat = {c: {"detection_rate" if c != BENIGN else "false_positive_rate":
                      round(float(if_flag[s.y_true == c].mean()), 4), "support": int((s.y_true == c).sum())}
                  for c in sorted(set(s.y_true))}
    routes, labels = s.hybrid(theta, tau)
    route_counts = {r: int((routes == r).sum()) for r in
                    (ROUTE_BELOW_THRESHOLD, ROUTE_CLEARED, ROUTE_CLASSIFIED, ROUTE_UNKNOWN)}
    return {
        "isolation_forest": {
            "binary": binary_metrics(attack, if_flag),
            "per_class": if_per_cat,
            "note": "IF is a one-class detector: per-class values are detection rates.",
        },
        "random_forest": {
            "binary": binary_metrics(attack, s.rf_class != BENIGN),
            **multiclass_metrics(s.y_true, s.rf_class),
        },
        "hybrid": {
            "binary": binary_metrics(attack, np.isin(routes, FLAGGED_ROUTES)),
            **multiclass_metrics(s.y_true, labels),
            "routes": route_counts,
            "unknown_anomaly_label": UNKNOWN_ANOMALY,
        },
        "theta": round(float(theta), 6),
        "tau": tau,
    }


def sweep(s: Scored, benign_val_scores, percentiles=(95, 97, 98, 99, 99.5),
          taus=(0.5, 0.6, 0.7, 0.8, 0.9), base_pct=99.0, base_tau=0.70) -> dict:
    """Hybrid sensitivity to theta (tau fixed) and tau (theta fixed)."""
    out = {"theta_percentile": {}, "tau": {}}
    attack = s.y_true != BENIGN
    for pct in percentiles:
        th = theta_from_scores(benign_val_scores, pct)
        routes, labels = s.hybrid(th, base_tau)
        mc = multiclass_metrics(s.y_true, labels)
        out["theta_percentile"][str(pct)] = {
            "theta": round(th, 6), "binary": binary_metrics(attack, np.isin(routes, FLAGGED_ROUTES)),
            "macro": mc["macro"], "if_benign_flag_rate": round(float((s.scores[~attack] >= th).mean()), 4),
            "recall_per_class": {c: v["recall"] for c, v in mc["per_class"].items()},
        }
    th = theta_from_scores(benign_val_scores, base_pct)
    for tau in taus:
        routes, labels = s.hybrid(th, tau)
        mc = multiclass_metrics(s.y_true, labels)
        out["tau"][str(tau)] = {
            "binary": binary_metrics(attack, np.isin(routes, FLAGGED_ROUTES)), "macro": mc["macro"],
            "unknown_anomalies": int((routes == ROUTE_UNKNOWN).sum()),
        }
    return out


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in FEATURE_COLUMNS if c in df.columns]]


def clean(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    out, stats = clean_rows(df, "Label")
    if verbose:
        print(f"  cleaning: {stats['dropped_non_finite']:,} non-finite and "
              f"{stats['dropped_duplicates']:,} duplicate rows dropped -> {stats['rows_out']:,} rows")
    out.attrs["clean_stats"] = stats
    return out
