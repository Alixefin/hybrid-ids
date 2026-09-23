"""Runtime-tunable settings (theta percentile, tau, enforcer mode).

Defaults come from config.py / environment; an admin can override them from the
"Model management" page. Overrides are stored in instance/runtime_settings.json so they
survive web-app reloads on PythonAnywhere.
"""
from __future__ import annotations

import json
from pathlib import Path

from flask import current_app

KEYS = {
    "theta_percentile": ("IF_THRESHOLD_PERCENTILE", float),
    "tau": ("RF_CONFIDENCE_TAU", float),
    "enforcer_mode": ("ENFORCER_MODE", str),
}


def _path() -> Path:
    return Path(current_app.config["INSTANCE_DIR"]) / "runtime_settings.json"


def _overrides() -> dict:
    p = _path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except ValueError:
            return {}
    return {}


def get(key: str):
    cfg_key, cast = KEYS[key]
    over = _overrides()
    return cast(over[key]) if key in over else cast(current_app.config[cfg_key])


def all_settings() -> dict:
    return {k: get(k) for k in KEYS}


def update(**values) -> dict:
    over = _overrides()
    if "theta_percentile" in values:
        v = float(values["theta_percentile"])
        if not 50.0 <= v <= 99.99:
            raise ValueError("theta percentile must be between 50 and 99.99")
        over["theta_percentile"] = v
    if "tau" in values:
        v = float(values["tau"])
        if not 0.0 < v <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        over["tau"] = v
    if "enforcer_mode" in values:
        v = str(values["enforcer_mode"])
        if v not in ("simulate", "enforce"):
            raise ValueError("mode must be 'simulate' or 'enforce'")
        over["enforcer_mode"] = v
    _path().write_text(json.dumps(over, indent=2))
    return all_settings()
