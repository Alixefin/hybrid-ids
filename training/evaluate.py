"""Offline evaluation - three protocols, three systems.

    python -m training.evaluate                     # all protocols on data/prepared/
    python -m training.evaluate --protocols A,C     # subset
    python -m training.evaluate --fast              # fewer trees (quick smoke test)

Protocols (on whatever source was prepared - synthetic or real):
  A  stratified random 70/15/15 train/val/test split
     (+ theta / tau sensitivity sweep, + ablation without the correlation filter)
  B  temporal: for each capture day the earliest 60 % of flows form train/val, the latest
     40 % the test set. Attack classes that only occur late in a day are absent from
     training - reported under "classes_absent_from_train".
  C  leave-one-attack-family-out: for each attack family F, train/val exclude F, the test
     set keeps it. Reports how the unseen family is handled (missed by the IF gate,
     cleared as benign, raised as "Unknown anomaly", or mislabelled as another attack).

Systems: Isolation Forest alone, Random Forest alone, hybrid pipeline. Metrics:
precision, recall, F1, false-positive rate - binary (attack vs benign), per class and
macro-averaged. Output: reports/metrics_<source>_<timestamp>.json and
reports/latest_metrics.json (shown on the dashboard's model page).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.detector import (  # noqa: E402
    ROUTE_BELOW_THRESHOLD, ROUTE_CLASSIFIED, ROUTE_CLEARED, ROUTE_UNKNOWN,
)
from engine.schema import BENIGN  # noqa: E402
from training.pipeline import (  # noqa: E402
    DEFAULT_PARAMS, clean, fit_pipeline, score, split_random, sweep, system_metrics,
)
from training.prepare_cicids2017 import SYNTHETIC_BANNER, load_prepared  # noqa: E402

REPORTS_DIR = ROOT / "reports"


def _params(fast: bool, corr_threshold: float | None = None) -> dict:
    p = json.loads(json.dumps(DEFAULT_PARAMS))
    if fast:
        p["random_forest"]["n_estimators"] = 40
        p["isolation_forest"]["n_estimators"] = 100
        p["preprocess"]["rf_estimators"] = 40
    if corr_threshold is not None:
        p["preprocess"]["corr_threshold"] = corr_threshold
    return p


def _summary_line(name, m):
    rows = []
    for sys_name in ("isolation_forest", "random_forest", "hybrid"):
        b = m[sys_name]["binary"]
        macro = m[sys_name].get("macro")
        extra = f"  macro-F1 {macro['f1']:.4f}" if macro else ""
        rows.append(f"    {name:4s} {sys_name:17s} P {b['precision']:.4f}  R {b['recall']:.4f}  "
                    f"F1 {b['f1']:.4f}  FPR {b['fpr']:.5f}{extra}")
    return "\n".join(rows)


def protocol_a(df, params, seed, ablation=True):
    print("\n[Protocol A] stratified random 70/15/15")
    train, val, test = split_random(df, seed=seed)
    fp = fit_pipeline(train, val, params, seed=seed)
    s = score(fp, test)
    res = {"split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
           "selected_features": fp.preprocessor.selected_features_,
           **system_metrics(s, fp.theta, params["tau"]),
           "sweep": sweep(s, fp.benign_val_scores, base_pct=params["theta_percentile"],
                          base_tau=params["tau"])}
    print(_summary_line("A", res))
    if ablation:
        print("  ablation: no correlation filter")
        p2 = json.loads(json.dumps(params))
        p2["preprocess"]["corr_threshold"] = 1.01
        fp2 = fit_pipeline(train, val, p2, seed=seed, verbose=False)
        m2 = system_metrics(score(fp2, test), fp2.theta, params["tau"])
        res["ablation_no_corr_filter"] = {
            k: {"binary": m2[k]["binary"], **({"macro": m2[k]["macro"]} if "macro" in m2[k] else {}),
                **({"per_class": m2[k]["per_class"]} if k == "isolation_forest" else {})}
            for k in ("isolation_forest", "random_forest", "hybrid")}
        print(_summary_line("A-nc", m2))
    return res


def temporal_split(df, early_share=0.6, val_share=0.15, seed=42):
    early_parts, late_parts = [], []
    for _, day in df.groupby("Day", sort=False):
        day = day.sort_values(["captured_at", "source_file", "row_in_file"], kind="stable")
        cut = int(round(len(day) * early_share))
        early_parts.append(day.iloc[:cut])
        late_parts.append(day.iloc[cut:])
    early, test = pd.concat(early_parts), pd.concat(late_parts)
    # train/val inside the early part (val only calibrates theta)
    counts = early["Category"].value_counts()
    strat = early["Category"] if counts.min() >= 2 else None
    from sklearn.model_selection import train_test_split
    train, val = train_test_split(early, test_size=val_share, stratify=strat, random_state=seed)
    return train, val, test


def protocol_b(df, params, seed):
    print("\n[Protocol B] temporal: earliest 60% of each day -> train/val, latest 40% -> test")
    train, val, test = temporal_split(df, seed=seed)
    absent = sorted(set(test["Category"]) - set(train["Category"]))
    if absent:
        print(f"  classes in test but absent from training: {absent}")
    fp = fit_pipeline(train, val, params, seed=seed)
    res = {"split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
           "classes_absent_from_train": absent,
           "test_class_counts": {k: int(v) for k, v in test["Category"].value_counts().items()},
           "train_class_counts": {k: int(v) for k, v in train["Category"].value_counts().items()},
           **system_metrics(score(fp, test), fp.theta, params["tau"])}
    print(_summary_line("B", res))
    return res


def protocol_c(df, params, seed):
    print("\n[Protocol C] leave-one-attack-family-out")
    families = sorted(c for c in df["Category"].unique() if c != BENIGN)
    train_all, val_all, test = split_random(df, seed=seed)
    folds = {}
    for fam in families:
        print(f"  held-out family: {fam}")
        train = train_all[train_all["Category"] != fam]
        val = val_all[val_all["Category"] != fam]
        fp = fit_pipeline(train, val, params, seed=seed, verbose=False)
        s = score(fp, test)
        m = system_metrics(s, fp.theta, params["tau"])
        routes, labels = s.hybrid(fp.theta, params["tau"])
        is_fam = s.y_true == fam
        n = int(is_fam.sum())
        r = routes[is_fam]
        mislabelled = (r == ROUTE_CLASSIFIED)
        outcome = {
            "support": n,
            "missed_by_if_gate": round(float((r == ROUTE_BELOW_THRESHOLD).mean()), 4),
            "cleared_as_benign_by_rf": round(float((r == ROUTE_CLEARED).mean()), 4),
            "raised_as_unknown_anomaly": round(float((r == ROUTE_UNKNOWN).mean()), 4),
            "flagged_as_other_attack": round(float(mislabelled.mean()), 4),
            "hybrid_detection_rate": round(float(np.isin(r, [ROUTE_UNKNOWN, ROUTE_CLASSIFIED]).mean()), 4),
            "if_alone_detection_rate": round(float((s.scores[is_fam] >= fp.theta).mean()), 4),
            "rf_alone_detection_rate": round(float((s.rf_class[is_fam] != BENIGN).mean()), 4),
            "other_attack_labels_used": {k: int(v) for k, v in
                                         pd.Series(labels[is_fam][mislabelled]).value_counts().items()},
        }
        print(f"    IF {outcome['if_alone_detection_rate']:.3f}  RF {outcome['rf_alone_detection_rate']:.3f}  "
              f"hybrid {outcome['hybrid_detection_rate']:.3f}  (unknown {outcome['raised_as_unknown_anomaly']:.3f}, "
              f"other-attack {outcome['flagged_as_other_attack']:.3f}, cleared-as-benign "
              f"{outcome['cleared_as_benign_by_rf']:.3f}, missed-by-gate {outcome['missed_by_if_gate']:.3f})")
        folds[fam] = {"held_out_outcome": outcome,
                      "overall": {k: {"binary": m[k]["binary"],
                                      **({"macro": m[k]["macro"]} if "macro" in m[k] else {})}
                                  for k in ("isolation_forest", "random_forest", "hybrid")}}
    mean = lambda key: round(float(np.mean([f["held_out_outcome"][key] for f in folds.values()])), 4)  # noqa: E731
    return {"folds": folds, "mean_over_families": {
        k: mean(k) for k in ("if_alone_detection_rate", "rf_alone_detection_rate",
                             "hybrid_detection_rate", "raised_as_unknown_anomaly", "cleared_as_benign_by_rf",
                             "missed_by_if_gate")}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocols", default="A,B,C")
    ap.add_argument("--fast", action="store_true", help="fewer trees for a quick run")
    ap.add_argument("--corr-threshold", type=float, default=None)
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    df, meta = load_prepared()
    synthetic = meta["data_source"] == "synthetic"
    if synthetic:
        print(SYNTHETIC_BANNER)
    df = clean(df)
    params = _params(args.fast, args.corr_threshold)
    protocols = [p.strip().upper() for p in args.protocols.split(",") if p.strip()]

    report = {
        "data_source": meta["data_source"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows_after_cleaning": int(len(df)),
        "cleaning": df.attrs.get("clean_stats", {}),
        "params": params,
        "protocols": {},
    }
    if synthetic:
        report["warning"] = ("SYNTHETIC DATA - these numbers only show that the pipeline runs; "
                             "they are NOT evaluation results for the thesis.")
    if meta.get("time_proxy_used"):
        report["note_temporal"] = "Timestamps missing in some files - row order used as time proxy."

    if "A" in protocols:
        report["protocols"]["A"] = protocol_a(df, params, args.seed, ablation=not args.no_ablation)
    if "B" in protocols:
        report["protocols"]["B"] = protocol_b(df, params, args.seed)
    if "C" in protocols:
        report["protocols"]["C"] = protocol_c(df, params, args.seed)

    REPORTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = REPORTS_DIR / f"metrics_{meta['data_source']}_{stamp}.json"
    text = json.dumps(report, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    out.write_text(text)
    (REPORTS_DIR / "latest_metrics.json").write_text(text)
    print(f"\nMetrics written to {out}")
    if synthetic:
        print(SYNTHETIC_BANNER)
    return report


if __name__ == "__main__":
    main()
