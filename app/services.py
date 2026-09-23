"""Service layer: runs the five-layer pipeline inside the Flask app.

    ingest (engine.ingest) -> detect (engine.detector) -> store flagged flows / detections
    -> policy (engine.policy) -> enforce (engine.enforcer) -> notify (engine.notifier)
    -> audit_log

Everything is synchronous: detection is triggered by the Offline Evaluation upload, the
admin "run detection on this file" button or ``flask run-detection``. The action-expiry
job (``expire_actions``) runs from the admin button, ``flask expire-actions``, a daily
PythonAnywhere scheduled task, or the optional APScheduler job.
"""
from __future__ import annotations

import json
import math
import threading
import time
import uuid
from datetime import timedelta
from pathlib import Path

import numpy as np
from flask import current_app

from app import audit, settings
from app.models import (
    Detection, FlowRecord, ModelVersion, ResponseAction, ResponsePolicy, Whitelist, db,
)
from engine.detector import (
    FLAGGED_ROUTES, ROUTE_BELOW_THRESHOLD, ROUTE_CLASSIFIED, ROUTE_CLEARED, ROUTE_UNKNOWN,
    HybridDetector,
)
from engine.enforcer import IptablesEnforcer, STATUS_FAILED
from engine.ingest import int_or_none, ip_str, read_flow_csv, to_datetime_or_now, utcnow
from engine.notifier import Notifier, SmtpSettings
from engine.policy import (
    ACTION_ALERT, ACTION_LABELS, BLOCKING_ACTIONS, DEFAULT_POLICY_TABLE, TIMED_ACTIONS,
    PolicyRule, plan_response,
)
from engine.schema import BENIGN, FEATURE_COLUMNS

_detector_cache: dict = {}
_detector_lock = threading.Lock()


class NoActiveModels(RuntimeError):
    pass


# =============================================================================== models
def register_models_from_store(activate_if_none: bool = True) -> list[str]:
    """Import every models_store/run_*.json manifest into model_version (idempotent)."""
    models_dir = Path(current_app.config["MODELS_DIR"])
    added = []
    for manifest_path in sorted(models_dir.glob("run_*.json")):
        m = json.loads(manifest_path.read_text())
        for kind in ("isolation_forest", "random_forest"):
            part = m[kind]
            if not (models_dir / part["file"]).exists():
                continue
            if ModelVersion.query.filter_by(file_path=part["file"]).first():
                continue
            params = {**part["params"], "run_id": m["run_id"], "data_source": m["data_source"],
                      "feature_names": m["feature_names"], "sklearn_version": m.get("sklearn_version")}
            metrics = {"binary": part["metrics"]} if kind == "isolation_forest" else part["metrics"]
            metrics["hybrid"] = m.get("hybrid_metrics", {})
            if kind == "isolation_forest":
                metrics["per_class"] = m["full_test_metrics"]["isolation_forest"]["per_class"]
            else:
                metrics["per_class"] = m["full_test_metrics"]["random_forest"]["per_class"]
            db.session.add(ModelVersion(
                model_type=kind, file_path=part["file"], params=params, metrics=metrics,
                is_active=False, trained_at=to_datetime_or_now(m.get("trained_at"))))
            added.append(part["file"])
    db.session.flush()
    if activate_if_none and not ModelVersion.query.filter_by(is_active=True).first():
        newest = ModelVersion.query.order_by(ModelVersion.trained_at.desc(), ModelVersion.model_id.desc()).first()
        if newest:
            activate_run(newest.run_id, user_id=False)
    if added:
        audit.record("models.registered", details={"files": added})
    db.session.commit()
    return added


def activate_run(run_id: str, user_id=None):
    """Activate the IF + RF pair of one training run (they share a feature set)."""
    models = [mv for mv in ModelVersion.query.all() if mv.run_id == run_id]
    types = {mv.model_type for mv in models}
    if types != {"isolation_forest", "random_forest"}:
        raise ValueError(f"run {run_id} does not have both an Isolation Forest and a Random Forest")
    ModelVersion.query.update({ModelVersion.is_active: False})
    for mv in models:
        mv.is_active = True
    audit.record("models.activated", user_id=user_id, details={"run_id": run_id})
    db.session.commit()
    clear_detector_cache()


def active_models() -> tuple[ModelVersion | None, ModelVersion | None]:
    q = ModelVersion.query.filter_by(is_active=True)
    return (q.filter_by(model_type="isolation_forest").first(),
            q.filter_by(model_type="random_forest").first())


def clear_detector_cache():
    with _detector_lock:
        _detector_cache.clear()


def get_detector() -> tuple[HybridDetector, ModelVersion, ModelVersion]:
    if_mv, rf_mv = active_models()
    if not if_mv or not rf_mv:
        raise NoActiveModels("No active models - train them (python -m training.train_models) "
                             "and register them on the Model management page.")
    theta_pct, tau = settings.get("theta_percentile"), settings.get("tau")
    key = (if_mv.model_id, rf_mv.model_id, theta_pct, tau)
    with _detector_lock:
        det = _detector_cache.get(key)
        if det is None:
            d = Path(current_app.config["MODELS_DIR"])
            det = HybridDetector.from_files(d / if_mv.file_path, d / rf_mv.file_path,
                                            tau=tau, theta_percentile=theta_pct)
            _detector_cache.clear()
            _detector_cache[key] = det
    return det, if_mv, rf_mv


# ============================================================================= helpers
def whitelist_entries() -> list[str]:
    return [w.ip_or_cidr for w in Whitelist.query.all()]


def policy_rules() -> list[PolicyRule]:
    return [PolicyRule(p.threat_class, p.severity, p.min_confidence, p.action_type,
                       p.duration_min, p.enabled, p.policy_id)
            for p in ResponsePolicy.query.all()]


def seed_default_policies(replace: bool = False) -> int:
    if replace:
        ResponsePolicy.query.delete()
    elif ResponsePolicy.query.first():
        return 0
    for r in DEFAULT_POLICY_TABLE:
        db.session.add(ResponsePolicy(threat_class=r.threat_class, severity=r.severity,
                                      min_confidence=r.min_confidence, action_type=r.action_type,
                                      duration_min=r.duration_min, enabled=r.enabled))
    return len(DEFAULT_POLICY_TABLE)


def make_enforcer() -> IptablesEnforcer:
    return IptablesEnforcer(mode=settings.get("enforcer_mode"), whitelist_provider=whitelist_entries)


def make_notifier() -> Notifier:
    c = current_app.config
    return Notifier(SmtpSettings(c["SMTP_HOST"], c["SMTP_PORT"], c["SMTP_USER"], c["SMTP_PASSWORD"],
                                 c["ALERT_EMAIL_TO"]))


def _json_safe(v):
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else ("Infinity" if f > 0 else "-Infinity" if f < 0 else None)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def prior_offences(ip: str, action_type: str, window_hours: int) -> int:
    """Earlier actions of the same type against ``ip`` inside the repeat-offence window
    (failed / not-executed actions do not count)."""
    since = utcnow() - timedelta(hours=window_hours)
    return (ResponseAction.query
            .filter(ResponseAction.target_ip == ip,
                    ResponseAction.action_type == action_type,
                    ResponseAction.status.in_(["active", "simulated", "expired", "released"]),
                    ResponseAction.executed_at >= since)
            .count())


def _live_action_keys() -> set[tuple[str, str]]:
    """(target, action_type) pairs that already have an unexpired blocking action."""
    now = utcnow()
    rows = (ResponseAction.query
            .filter(ResponseAction.status.in_(["active", "simulated"]),
                    ResponseAction.action_type.in_(list(BLOCKING_ACTIONS)))
            .all())
    return {(a.target_ip, a.action_type) for a in rows if a.expires_at is None or a.expires_at > now}


def execute_action(planned, detection: Detection | None, enforcer, approved_by=None,
                   source="policy") -> ResponseAction:
    """Run one planned action through the enforcer and persist + audit it."""
    now = utcnow()
    tag = uuid.uuid4().hex[:12]
    if planned.skipped_reason:
        status, outcome, expires = STATUS_FAILED, f"NOT EXECUTED - {planned.skipped_reason}", None
    else:
        result = enforcer.apply(planned.action_type, planned.target_ip, tag)
        status = result.status
        if planned.action_type == ACTION_ALERT:
            status = "active"
        outcome = result.output
        expires = (now + timedelta(minutes=planned.duration_min)
                   if result.success and planned.action_type in TIMED_ACTIONS and planned.duration_min > 0
                   else None)
    action = ResponseAction(
        detection_id=detection.detection_id if detection else None, policy_id=planned.policy_id,
        approved_by=approved_by, action_type=planned.action_type, target_ip=planned.target_ip,
        mode=enforcer.mode, status=status, executed_at=now, expires_at=expires,
        outcome=f"[tag {tag}] {planned.reason}\n{outcome}".strip())
    db.session.add(action)
    db.session.flush()
    audit.record(f"response.{planned.action_type}", user_id=approved_by if approved_by else False,
                 details={"action_id": action.action_id, "target": planned.target_ip,
                          "mode": enforcer.mode, "status": status, "source": source,
                          "detection_id": action.detection_id, "duration_min": planned.duration_min,
                          "skipped": planned.skipped_reason})
    return action


# =========================================================================== detection
def run_detection(path, source_name: str | None = None, persist: bool = True,
                  flow_source: str = "csv", user_id=None) -> dict:
    """Score a CICFlowMeter CSV synchronously and (optionally) store + respond.

    Returns a JSON-serialisable summary, also saved to instance/runs/<run_id>.json.
    """
    t0 = time.time()
    cfg = current_app.config
    detector, if_mv, rf_mv = get_detector()
    batch = read_flow_csv(path, source_name=source_name, max_rows=cfg["MAX_UPLOAD_ROWS"])
    if len(batch) == 0:
        raise ValueError("the file contains no flow rows")
    result = detector.detect(batch.features, batch_size=cfg["DETECTION_BATCH_SIZE"])
    frame = result.frame
    t_detect = time.time() - t0

    run_id = uuid.uuid4().hex[:12]
    summary = {
        "run_id": run_id, "file": batch.source_name, "persisted": persist,
        "started_at": utcnow().isoformat(timespec="seconds"),
        "rows_read": batch.rows_read, "rows_scored": len(batch),
        "rows_dropped_empty": batch.rows_dropped_empty, "notes": batch.notes,
        "counts": result.counts(), "theta": round(result.theta, 6),
        "theta_percentile": detector.theta_percentile, "tau": detector.tau,
        "models": {"run_id": if_mv.run_id, "if_model_id": if_mv.model_id, "rf_model_id": rf_mv.model_id},
        "data_source": detector.data_source, "enforcer_mode": settings.get("enforcer_mode"),
    }
    flagged = frame[frame["route"].isin(FLAGGED_ROUTES)]
    summary["flagged_by_class"] = {k: int(v) for k, v in flagged["label"].value_counts().items()}
    top = (flagged.assign(src=batch.ids.loc[flagged.index, "Source IP"].map(ip_str))
           .groupby("src").size().sort_values(ascending=False).head(10))
    summary["top_sources"] = {k or "unknown": int(v) for k, v in top.items()}

    if batch.has_labels:
        summary["ground_truth"] = _score_against_labels(batch.categories, frame)

    actions_summary = {"executed": {}, "suppressed_duplicates": 0, "skipped": 0, "alerts_sent": 0}
    if persist:
        actions_summary = _persist_and_respond(batch, frame, if_mv, rf_mv, flow_source)
    summary["actions"] = actions_summary
    summary["elapsed_s"] = {"detection": round(t_detect, 2), "total": round(time.time() - t0, 2)}

    audit.record("detection_run", user_id=user_id, details={
        "run_id": run_id, "file": batch.source_name, "total": len(batch),
        "flagged": int(len(flagged)), "persisted": persist, **summary["counts"],
        "data_source": detector.data_source})
    db.session.commit()
    _save_run_summary(summary)
    return summary


def _score_against_labels(categories, frame) -> dict:
    from training.pipeline import binary_metrics, multiclass_metrics  # pure functions
    y = categories.loc[frame.index].to_numpy(dtype=object)
    labels = frame["label"].to_numpy(dtype=object)
    attack = y != BENIGN
    return {
        "hybrid_binary": binary_metrics(attack, frame["route"].isin(FLAGGED_ROUTES).to_numpy()),
        "if_binary": binary_metrics(attack, (frame["route"] != ROUTE_BELOW_THRESHOLD).to_numpy()),
        "hybrid_multiclass": multiclass_metrics(y, labels),
        "true_class_counts": {k: int(v) for k, v in categories.value_counts().items()},
    }


def _persist_and_respond(batch, frame, if_mv, rf_mv, flow_source) -> dict:
    cfg = current_app.config
    stored = frame[frame["route"] != ROUTE_BELOW_THRESHOLD]  # benign below theta: counted only
    feat_cols = [c for c in FEATURE_COLUMNS if c in batch.features.columns]
    rules, whitelist = policy_rules(), whitelist_entries()
    enforcer, notifier = make_enforcer(), make_notifier()
    live_keys = _live_action_keys()
    offence_cache: dict[tuple[str, str], int] = {}
    alert_groups: dict[tuple[str, str], dict] = {}
    out = {"executed": {}, "suppressed_duplicates": 0, "skipped": 0, "alerts_sent": 0}

    def offences(ip, action_type):
        key = (ip, action_type)
        if key not in offence_cache:
            offence_cache[key] = prior_offences(ip, action_type, cfg["REPEAT_OFFENCE_WINDOW_HOURS"])
        return offence_cache[key]

    for idx, row in stored.iterrows():
        ids = batch.ids.loc[idx]
        feats = {c: _json_safe(v) for c, v in batch.features.loc[idx, feat_cols].items()}
        feats["_meta"] = {
            "flow_id": ip_str(ids.get("Flow ID")), "timestamp_raw": ids.get("Timestamp", ""),
            "route": row["route"], "rf_class": row["rf_class"], "repaired": bool(row["repaired"]),
            "upload": batch.source_name, "row": int(idx),
            "ground_truth": batch.categories.loc[idx] if batch.has_labels else None,
            "ground_truth_label": batch.labels.loc[idx] if batch.has_labels else None,
        }
        flow = FlowRecord(src_ip=ip_str(ids.get("Source IP")), dst_ip=ip_str(ids.get("Destination IP")),
                          src_port=int_or_none(ids.get("Source Port")),
                          dst_port=int_or_none(ids.get("Destination Port")),
                          protocol=int_or_none(ids.get("Protocol")), features=feats,
                          captured_at=to_datetime_or_now(ids.get("captured_at")), source=flow_source)
        db.session.add(flow)
        db.session.flush()
        cleared = row["route"] == ROUTE_CLEARED
        det = Detection(flow_id=flow.flow_id, if_model_id=if_mv.model_id, rf_model_id=rf_mv.model_id,
                        anomaly_score=float(row["anomaly_score"]), predicted_class=row["label"],
                        confidence=None if np.isnan(row["confidence"]) else float(row["confidence"]),
                        severity=row["severity"], status="resolved" if cleared else "new")
        db.session.add(det)
        db.session.flush()
        if cleared:
            continue

        decision = plan_response(rules, row["label"], float(row["confidence"]), flow.src_ip,
                                 flow.dst_ip, whitelist, offences, severity=row["severity"],
                                 cap_min=cfg["MAX_BLOCK_MINUTES"])
        for planned in decision.actions:
            if planned.action_type == ACTION_ALERT:
                g = alert_groups.setdefault((row["label"], planned.target_ip), {
                    "detection": det, "planned": planned, "count": 0, "max_conf": 0.0})
                g["count"] += 1
                g["max_conf"] = max(g["max_conf"], float(row["confidence"]))
                continue
            key = (planned.target_ip, planned.action_type)
            if planned.skipped_reason is None and key in live_keys:
                out["suppressed_duplicates"] += 1
                continue
            action = execute_action(planned, det, enforcer)
            if planned.skipped_reason:
                out["skipped"] += 1
            else:
                live_keys.add(key)
                if planned.action_type in TIMED_ACTIONS:
                    offence_cache[key] = offences(*key) + 1
            out["executed"][action.action_type] = out["executed"].get(action.action_type, 0) + 1

    # One SOC alert per (class, target) per run instead of one per flow.
    for (label, target), g in alert_groups.items():
        planned = g["planned"]
        planned.reason = (f"{label}: {g['count']} flow(s) involving {target} in {batch.source_name}, "
                          f"max confidence {g['max_conf']:.2f}")
        channels, err = notifier.send(
            f"{label} ({planned.severity}) - {target}",
            f"{planned.reason}\nDetection #{g['detection'].detection_id}. "
            f"Mode: {enforcer.mode}.", urgent=planned.urgent)
        action = execute_action(planned, g["detection"], enforcer)
        action.outcome += f"\nnotified via: {', '.join(channels)}" + (f" ({err})" if err else "")
        out["alerts_sent"] += 1
        out["executed"][ACTION_ALERT] = out["executed"].get(ACTION_ALERT, 0) + 1
    return out


def _runs_dir() -> Path:
    p = Path(current_app.config["INSTANCE_DIR"]) / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_run_summary(summary: dict):
    (_runs_dir() / f"{summary['run_id']}.json").write_text(json.dumps(summary, indent=2, default=str))


def load_run_summary(run_id: str) -> dict | None:
    if not run_id.isalnum():
        return None
    p = _runs_dir() / f"{run_id}.json"
    return json.loads(p.read_text()) if p.exists() else None


# ========================================================================= responses
def expire_actions(user_id=None) -> int:
    """Release every active / simulated action whose expiry time has passed."""
    now = utcnow()
    due = (ResponseAction.query
           .filter(ResponseAction.status.in_(["active", "simulated"]),
                   ResponseAction.expires_at.isnot(None), ResponseAction.expires_at <= now)
           .all())
    enforcer = make_enforcer()
    for a in due:
        result = enforcer.release(a.action_type, a.target_ip, _tag_of(a))
        a.status = "expired" if result.success else "failed"
        a.outcome = (a.outcome or "") + f"\n[{now:%Y-%m-%d %H:%M}] expired: {result.output}"
        audit.record("response.expired", user_id=user_id if user_id else False,
                     details={"action_id": a.action_id, "target": a.target_ip,
                              "action_type": a.action_type, "ok": result.success})
    audit.record("job.expire_actions", user_id=user_id if user_id else False,
                 details={"expired": len(due)})
    db.session.commit()
    return len(due)


def release_action(action: ResponseAction, user_id) -> ResponseAction:
    if action.status not in ("active", "simulated"):
        raise ValueError(f"action #{action.action_id} is {action.status}, not active")
    result = make_enforcer().release(action.action_type, action.target_ip, _tag_of(action))
    action.status = "released" if result.success else "failed"
    action.outcome = (action.outcome or "") + f"\n[{utcnow():%Y-%m-%d %H:%M}] released manually: {result.output}"
    audit.record("response.released", user_id=user_id,
                 details={"action_id": action.action_id, "target": action.target_ip, "ok": result.success})
    db.session.commit()
    return action


def manual_action(detection: Detection, action_type: str, target_ip: str, duration_min: int,
                  user_id) -> ResponseAction:
    """Analyst-approved action (e.g. blocking the source of an Unknown anomaly)."""
    from engine.policy import PlannedAction, match_whitelist, parse_ip
    if action_type not in ACTION_LABELS:
        raise ValueError("unknown action type")
    try:
        parse_ip(target_ip)
    except ValueError as exc:
        raise ValueError(f"invalid target IP {target_ip!r}") from exc
    wl = match_whitelist(target_ip, whitelist_entries()) if action_type in BLOCKING_ACTIONS else None
    planned = PlannedAction(action_type, target_ip, max(0, int(duration_min)),
                            detection.severity or "Medium", None,
                            reason=f"manual {ACTION_LABELS[action_type]} approved by user #{user_id}",
                            skipped_reason=f"target {target_ip} is whitelisted ({wl})" if wl else None)
    action = execute_action(planned, detection, make_enforcer(), approved_by=user_id, source="manual")
    db.session.commit()
    return action


def _tag_of(action: ResponseAction) -> str:
    text = action.outcome or ""
    if text.startswith("[tag "):
        return text[5:text.index("]")]
    return str(action.action_id)
