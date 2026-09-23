"""Shared fixtures.

``tiny_run`` trains a small but real model set (IF + RF + pre-processor) once per test
session on a few thousand synthetic flows and saves it like train_models.py does, so
the detector and the Flask flow are tested against genuine artifacts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.ingest import split_frame, standardise_columns  # noqa: E402
from training import generate_synthetic_flows as gen  # noqa: E402

TINY_PARAMS = {
    "preprocess": {"n_features": 20, "corr_threshold": 0.95, "clip_quantile": 0.999,
                   "near_constant_share": 0.9995, "rf_estimators": 20, "max_fit_rows": 50_000},
    "isolation_forest": {"n_estimators": 50, "max_samples": 512, "max_features": 1.0, "bootstrap": False},
    "random_forest": {"n_estimators": 25, "max_depth": 20, "min_samples_leaf": 1,
                      "max_features": "sqrt", "class_weight": "balanced"},
    "theta_percentile": 99.0,
    "tau": 0.70,
    "max_benign_val_scores": 5000,
}


def synthetic_frame(rows=4000, seed=3) -> pd.DataFrame:
    """Synthetic flows in the same shape prepare_cicids2017 produces."""
    raw = gen.to_raw_csv_frame(gen.generate(rows, seed=seed, min_per_class=30))
    batch = split_frame(standardise_columns(raw), "synthetic")
    df = pd.concat([batch.ids.drop(columns=["Destination Port"]), batch.features], axis=1)
    df["Label"] = batch.labels.to_numpy()
    df["Category"] = batch.categories.to_numpy()
    return df


@pytest.fixture(scope="session")
def synthetic_df():
    return synthetic_frame()


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory, synthetic_df):
    from training.pipeline import clean, fit_pipeline, score, split_random, system_metrics
    from training.train_models import save_run

    models_dir = tmp_path_factory.mktemp("models")
    df = clean(synthetic_df, verbose=False)
    train, val, test = split_random(df, seed=1)
    fp = fit_pipeline(train, val, TINY_PARAMS, seed=1, verbose=False)
    metrics = system_metrics(score(fp, test), fp.theta, TINY_PARAMS["tau"])
    manifest = save_run(fp, metrics, {"data_source": "synthetic"}, "test-run", models_dir)
    return {"dir": models_dir, "manifest": manifest, "fp": fp, "test": test}


@pytest.fixture
def app(tmp_path, tiny_run):
    from app import create_app
    from app.models import db
    from config import TestConfig

    class Cfg(TestConfig):
        INSTANCE_DIR = tmp_path / "instance"
        UPLOAD_DIR = tmp_path / "uploads"
        MODELS_DIR = tiny_run["dir"]

    application = create_app(Cfg)
    with application.app_context():
        db.create_all()
        from app import services
        services.seed_default_policies()
        db.session.commit()
        services.clear_detector_cache()
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


def make_user(username="admin", password="correct-horse-1", role="admin"):
    from app.models import User, db
    u = User(username=username, role=role)
    u.set_password(password)
    db.session.add(u)
    db.session.commit()
    return u


@pytest.fixture
def rng():
    return np.random.default_rng(0)
