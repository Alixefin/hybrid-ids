"""Application configuration.

Every setting can be overridden with an environment variable of the same name.
On PythonAnywhere set them in the WSGI file (see DEPLOY.md) - the defaults already keep
the SQLite database, uploads and models inside the project directory, which lives on
PythonAnywhere's persistent storage (/home/<username>/...), never under /tmp.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _env(name: str, default):
    value = os.environ.get(name)
    if value is None:
        return default
    if isinstance(default, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def _secret_key(instance_dir: Path) -> str:
    """SECRET_KEY from the environment, else a random key generated once and kept in
    instance/secret_key (git-ignored), so no secret ever has to be written by hand."""
    if os.environ.get("SECRET_KEY"):
        return os.environ["SECRET_KEY"]
    path = instance_dir / "secret_key"
    try:
        if path.exists():
            return path.read_text().strip()
        instance_dir.mkdir(parents=True, exist_ok=True)
        key = secrets.token_hex(32)
        path.write_text(key)
        os.chmod(path, 0o600)
        return key
    except OSError:
        return secrets.token_hex(32)  # read-only filesystem: per-process key


class Config:
    # --- Paths (all persistent, relative to the project) ---------------------------
    BASE_DIR = BASE_DIR
    INSTANCE_DIR = Path(_env("HYBRID_IDS_INSTANCE_DIR", str(BASE_DIR / "instance")))
    DATA_DIR = Path(_env("HYBRID_IDS_DATA_DIR", str(BASE_DIR / "data")))
    MODELS_DIR = Path(_env("HYBRID_IDS_MODELS_DIR", str(BASE_DIR / "models_store")))
    UPLOAD_DIR = Path(_env("HYBRID_IDS_UPLOAD_DIR", str(BASE_DIR / "data" / "uploads")))

    # --- Flask -----------------------------------------------------------------------
    SECRET_KEY = _secret_key(INSTANCE_DIR)
    SQLALCHEMY_DATABASE_URI = _env(
        "DATABASE_URL", "sqlite:///" + str((INSTANCE_DIR / "hybrid_ids.db").as_posix())
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = _env("MAX_UPLOAD_MB", 50) * 1024 * 1024
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = _env("SESSION_COOKIE_SECURE", False)

    # --- Authentication --------------------------------------------------------------
    MAX_FAILED_LOGINS = _env("MAX_FAILED_LOGINS", 3)

    # --- Detection engine ------------------------------------------------------------
    # theta: anomaly-score threshold = this percentile of benign *validation* scores.
    IF_THRESHOLD_PERCENTILE = _env("IF_THRESHOLD_PERCENTILE", 99.0)
    # tau: Random Forest confidence threshold.
    RF_CONFIDENCE_TAU = _env("RF_CONFIDENCE_TAU", 0.70)
    # Rows scored per batch when evaluating an uploaded CSV (keeps memory bounded on
    # PythonAnywhere's free tier).
    DETECTION_BATCH_SIZE = _env("DETECTION_BATCH_SIZE", 20000)
    # Hard cap on rows accepted from a single upload, to stay inside the free tier's
    # request-time limits.
    MAX_UPLOAD_ROWS = _env("MAX_UPLOAD_ROWS", 200000)

    # --- Response layer --------------------------------------------------------------
    # "simulate" logs the iptables/conntrack command; "enforce" executes it (root on a
    # lab VM only - impossible on PythonAnywhere).
    ENFORCER_MODE = _env("ENFORCER_MODE", "simulate")
    REPEAT_OFFENCE_WINDOW_HOURS = _env("REPEAT_OFFENCE_WINDOW_HOURS", 24)
    MAX_BLOCK_MINUTES = _env("MAX_BLOCK_MINUTES", 24 * 60)

    # --- Optional background scheduler (action-expiry job) ---------------------------
    # Off by default: PythonAnywhere's free tier cannot run a persistent scheduler.
    # The same job is always available on demand from the admin "Run expiry now" button.
    SCHEDULER_ENABLED = _env("SCHEDULER_ENABLED", False)
    EXPIRY_JOB_INTERVAL_MIN = _env("EXPIRY_JOB_INTERVAL_MIN", 5)

    # --- Notifications ---------------------------------------------------------------
    # Optional SMTP; when unset, SOC alerts are written to the log + audit_log only.
    SMTP_HOST = _env("SMTP_HOST", "")
    SMTP_PORT = _env("SMTP_PORT", 587)
    SMTP_USER = _env("SMTP_USER", "")
    SMTP_PASSWORD = _env("SMTP_PASSWORD", "")
    ALERT_EMAIL_TO = _env("ALERT_EMAIL_TO", "")


class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    CSRF_ENABLED = False
    SECRET_KEY = "test"
    ENFORCER_MODE = "simulate"
    SCHEDULER_ENABLED = False
