"""Train the production models (Protocol A split) and save them to models_store/.

    python -m training.train_models                 # uses data/prepared/ (see prepare_cicids2017)
    python -m training.train_models --tau 0.7 --theta-percentile 99

Writes, for one training run <run_id>:
    models_store/if_<run_id>.joblib     Isolation Forest + fitted pre-processor + theta
    models_store/rf_<run_id>.joblib     Random Forest (class-weighted)
    models_store/run_<run_id>.json      manifest: params, metrics, data source
The Flask app imports manifests into the model_version table ("Model management" page
or `flask register-models`).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import sklearn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.pipeline import (  # noqa: E402
    DEFAULT_PARAMS, clean, fit_pipeline, score, split_random, system_metrics,
)
from training.prepare_cicids2017 import SYNTHETIC_BANNER, load_prepared  # noqa: E402

MODELS_DIR = ROOT / "models_store"


def save_run(fp, metrics: dict, meta: dict, run_id: str, models_dir: Path = MODELS_DIR) -> dict:
    models_dir.mkdir(parents=True, exist_ok=True)
    source = meta["data_source"]
    features = list(fp.preprocessor.selected_features_)
    trained_at = datetime.now().isoformat(timespec="seconds")
    common = {"run_id": run_id, "feature_names": features, "data_source": source,
              "trained_at": trained_at, "sklearn_version": sklearn.__version__}

    if_file, rf_file = f"if_{run_id}.joblib", f"rf_{run_id}.joblib"
    joblib.dump({**common, "kind": "isolation_forest", "model": fp.iforest,
                 "preprocessor": fp.preprocessor, "theta_percentile": fp.theta_percentile,
                 "theta": fp.theta, "benign_val_scores": fp.benign_val_scores,
                 "params": fp.params["isolation_forest"]},
                models_dir / if_file, compress=3)
    joblib.dump({**common, "kind": "random_forest", "model": fp.rf,
                 "classes": list(fp.rf.classes_), "params": fp.params["random_forest"]},
                models_dir / rf_file, compress=3)

    manifest = {
        **common,
        "isolation_forest": {"file": if_file, "params": {**fp.params["isolation_forest"],
                             "theta_percentile": fp.theta_percentile, "theta": round(fp.theta, 6)},
                             "metrics": metrics["isolation_forest"]["binary"]},
        "random_forest": {"file": rf_file, "params": fp.params["random_forest"],
                          "classes": list(fp.rf.classes_),
                          "metrics": {"binary": metrics["random_forest"]["binary"],
                                      "macro": metrics["random_forest"]["macro"]}},
        "hybrid_metrics": {"binary": metrics["hybrid"]["binary"], "macro": metrics["hybrid"]["macro"],
                           "routes": metrics["hybrid"]["routes"]},
        "tau": fp.params["tau"],
        "preprocessing": fp.preprocessor.report(),
        "train_counts": fp.train_counts,
        "timings": fp.timings,
        "full_test_metrics": metrics,
    }
    if source == "synthetic":
        manifest["warning"] = "Trained on SYNTHETIC data - not a real evaluation."
    (models_dir / f"run_{run_id}.json").write_text(json.dumps(manifest, indent=2, default=float))
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--theta-percentile", type=float, default=DEFAULT_PARAMS["theta_percentile"])
    ap.add_argument("--tau", type=float, default=DEFAULT_PARAMS["tau"])
    ap.add_argument("--n-features", type=int, default=30)
    ap.add_argument("--rf-trees", type=int, default=100)
    ap.add_argument("--corr-threshold", type=float, default=DEFAULT_PARAMS["preprocess"]["corr_threshold"],
                    help="|r| above which a feature is dropped (spec: 0.95; >1 disables the filter)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    df, meta = load_prepared()
    synthetic = meta["data_source"] == "synthetic"
    if synthetic:
        print(SYNTHETIC_BANNER)
    print(f"Training on data_source='{meta['data_source']}' ({len(df):,} prepared flows)")

    df = clean(df)
    train, val, test = split_random(df, seed=args.seed)
    print(f"  Protocol A split: train {len(train):,} / val {len(val):,} / test {len(test):,}")

    params = {**DEFAULT_PARAMS, "theta_percentile": args.theta_percentile, "tau": args.tau,
              "preprocess": {**DEFAULT_PARAMS["preprocess"], "n_features": args.n_features,
                             "corr_threshold": args.corr_threshold},
              "random_forest": {**DEFAULT_PARAMS["random_forest"], "n_estimators": args.rf_trees}}
    fp = fit_pipeline(train, val, params, seed=args.seed)
    metrics = system_metrics(score(fp, test), fp.theta, args.tau)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{meta['data_source']}"
    manifest = save_run(fp, metrics, meta, run_id)

    print(f"\nSaved run {run_id} -> {MODELS_DIR}")
    print(f"  selected features ({len(manifest['feature_names'])}): {', '.join(manifest['feature_names'])}")
    print("\n  Test-set results (Protocol A hold-out)")
    print(f"  {'system':18s} {'precision':>9s} {'recall':>7s} {'F1':>6s} {'FPR':>8s}   (binary: attack vs benign)")
    for name in ("isolation_forest", "random_forest", "hybrid"):
        b = metrics[name]["binary"]
        print(f"  {name:18s} {b['precision']:9.4f} {b['recall']:7.4f} {b['f1']:6.4f} {b['fpr']:8.5f}")
    for name in ("random_forest", "hybrid"):
        m = metrics[name]["macro"]
        print(f"  {name + ' macro':18s} {m['precision']:9.4f} {m['recall']:7.4f} {m['f1']:6.4f} {m['fpr']:8.5f}")
    print(f"  hybrid routing: {metrics['hybrid']['routes']}")
    if synthetic:
        print(SYNTHETIC_BANNER)
    return manifest


if __name__ == "__main__":
    main()
