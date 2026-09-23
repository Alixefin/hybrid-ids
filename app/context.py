"""Template context: synthetic-data banner, active models, labels, filters."""
from __future__ import annotations

from app import settings
from engine.policy import ACTION_LABELS
from engine.schema import THREAT_CATEGORIES, UNKNOWN_ANOMALY

SEVERITY_BADGE = {"Critical": "danger", "High": "danger", "Medium-High": "warning",
                  "Medium": "warning", "Low": "secondary", None: "light"}
STATUS_BADGE = {"new": "danger", "acknowledged": "warning", "resolved": "success",
                "false_positive": "secondary", "active": "danger", "simulated": "info",
                "expired": "secondary", "released": "success", "failed": "dark"}


def register(app):
    @app.context_processor
    def inject():
        from flask_login import current_user
        if not current_user or not current_user.is_authenticated:
            return {"action_labels": ACTION_LABELS}
        from app.services import active_models
        try:
            if_mv, rf_mv = active_models()
        except Exception:  # noqa: BLE001 - e.g. tables not created yet
            if_mv = rf_mv = None
        return {
            "active_if": if_mv, "active_rf": rf_mv,
            "data_source": if_mv.data_source if if_mv else None,
            "runtime": settings.all_settings(),
            "action_labels": ACTION_LABELS,
            "threat_categories": THREAT_CATEGORIES + [UNKNOWN_ANOMALY],
            "severity_badge": SEVERITY_BADGE, "status_badge": STATUS_BADGE,
        }

    @app.template_filter("dt")
    def fmt_dt(value, fmt="%Y-%m-%d %H:%M:%S"):
        return value.strftime(fmt) if value else "-"

    @app.template_filter("pct")
    def fmt_pct(value, digits=1):
        return "-" if value is None else f"{float(value) * 100:.{digits}f}%"
