from datetime import datetime, timedelta

from flask import abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from sqlalchemy import func

from app import audit
from app.audit import parse_details
from app.dashboard import bp
from app.models import DETECTION_STATUSES, AuditLog, Detection, FlowRecord, ResponseAction, db
from engine.ingest import utcnow
from engine.policy import ACTION_TYPES, BLOCKING_ACTIONS
from engine.schema import BENIGN

PER_PAGE = 50


def _run_totals():
    """Flows analysed across all detection runs. Benign flows below theta are only
    counted, never stored, so the totals come from the detection_run audit events."""
    keys = ("total", "below_threshold", "cleared_false_alarm", "classified_attack", "unknown_anomaly")
    totals = {"runs": 0, **{k: 0 for k in keys}}
    for r in AuditLog.query.filter(AuditLog.event.like("detection_run %")).all():
        _, d = parse_details(r.event)
        totals["runs"] += 1
        for k in keys:
            totals[k] += int(d.get(k, 0) or 0)
    return totals


@bp.route("/")
@login_required
def index():
    totals = _run_totals()
    flagged = Detection.query.filter(Detection.predicted_class != BENIGN)
    by_class = dict(db.session.query(Detection.predicted_class, func.count())
                    .filter(Detection.predicted_class != BENIGN)
                    .group_by(Detection.predicted_class).all())
    by_status = dict(db.session.query(Detection.status, func.count())
                     .filter(Detection.predicted_class != BENIGN).group_by(Detection.status).all())
    by_action = dict(db.session.query(ResponseAction.action_type, func.count())
                     .group_by(ResponseAction.action_type).all())
    now = utcnow()
    active_blocks = (ResponseAction.query
                     .filter(ResponseAction.status.in_(["active", "simulated"]),
                             ResponseAction.action_type.in_(list(BLOCKING_ACTIONS)))
                     .filter((ResponseAction.expires_at.is_(None)) | (ResponseAction.expires_at > now))
                     .count())
    day = func.date(FlowRecord.captured_at)
    per_day = (db.session.query(day, func.count())
               .join(Detection, Detection.flow_id == FlowRecord.flow_id)
               .filter(Detection.predicted_class != BENIGN)
               .group_by(day).order_by(day.desc()).limit(30).all())[::-1]
    top_sources = (db.session.query(FlowRecord.src_ip, func.count())
                   .join(Detection, Detection.flow_id == FlowRecord.flow_id)
                   .filter(Detection.predicted_class != BENIGN)
                   .group_by(FlowRecord.src_ip).order_by(func.count().desc()).limit(8).all())
    recent = flagged.order_by(Detection.detection_id.desc()).limit(10).all()
    charts = {
        "by_class": {"labels": list(by_class), "values": list(by_class.values())},
        "per_day": {"labels": [str(d) for d, _ in per_day], "values": [c for _, c in per_day]},
        "by_action": {"labels": list(by_action), "values": list(by_action.values())},
        "routes": {"labels": ["Below θ (benign)", "Cleared false alarm", "Classified attack",
                              "Unknown anomaly"],
                   "values": [totals["below_threshold"], totals["cleared_false_alarm"],
                              totals["classified_attack"], totals["unknown_anomaly"]]},
    }
    return render_template("dashboard/index.html", totals=totals, flagged_count=flagged.count(),
                           new_count=by_status.get("new", 0), active_blocks=active_blocks,
                           unknown_count=by_class.get("Unknown anomaly", 0), recent=recent,
                           top_sources=top_sources, charts=charts)


def _parse_date(value, end=False):
    try:
        d = datetime.strptime(value, "%Y-%m-%d")
        return d + timedelta(days=1) if end else d
    except (TypeError, ValueError):
        return None


@bp.route("/alerts")
@login_required
def alerts():
    f = {k: request.args.get(k, "").strip() for k in
         ("category", "severity", "status", "ip", "min_conf", "date_from", "date_to", "show_cleared")}
    q = Detection.query.join(FlowRecord, Detection.flow_id == FlowRecord.flow_id)
    if not f["show_cleared"]:
        q = q.filter(Detection.predicted_class != BENIGN)
    if f["category"]:
        q = q.filter(Detection.predicted_class == f["category"])
    if f["severity"]:
        q = q.filter(Detection.severity == f["severity"])
    if f["status"] in DETECTION_STATUSES:
        q = q.filter(Detection.status == f["status"])
    if f["ip"]:
        like = f"%{f['ip']}%"
        q = q.filter((FlowRecord.src_ip.like(like)) | (FlowRecord.dst_ip.like(like)))
    if f["min_conf"]:
        try:
            q = q.filter(Detection.confidence >= float(f["min_conf"]))
        except ValueError:
            flash("Minimum confidence must be a number between 0 and 1.", "warning")
    d_from, d_to = _parse_date(f["date_from"]), _parse_date(f["date_to"], end=True)
    if d_from:
        q = q.filter(Detection.detected_at >= d_from)
    if d_to:
        q = q.filter(Detection.detected_at < d_to)
    page = max(1, request.args.get("page", 1, type=int))
    pagination = q.order_by(Detection.detection_id.desc()).paginate(
        page=page, per_page=PER_PAGE, error_out=False)
    severities = sorted(s for (s,) in db.session.query(Detection.severity).distinct() if s)
    args = {k: v for k, v in f.items() if v}
    return render_template("dashboard/alerts.html", pagination=pagination, f=f, args=args,
                           statuses=DETECTION_STATUSES, severities=severities)


@bp.route("/alerts/<int:detection_id>")
@login_required
def alert_detail(detection_id):
    det = db.session.get(Detection, detection_id) or abort(404)
    feats = dict(det.flow.features or {})
    meta = feats.pop("_meta", {})
    related = (Detection.query.join(FlowRecord, Detection.flow_id == FlowRecord.flow_id)
               .filter(FlowRecord.src_ip == det.flow.src_ip, Detection.detection_id != det.detection_id,
                       Detection.predicted_class != BENIGN)
               .order_by(Detection.detection_id.desc()).limit(10).all())
    selected = (det.if_model.params or {}).get("feature_names", []) if det.if_model else []
    theta = None
    if det.if_model:
        theta = (det.if_model.params or {}).get("theta")
    return render_template("dashboard/alert_detail.html", d=det, features=feats, meta=meta,
                           related=related, selected=set(selected), statuses=DETECTION_STATUSES,
                           action_types=ACTION_TYPES, model_theta=theta)


@bp.route("/alerts/<int:detection_id>/status", methods=["POST"])
@login_required
def set_status(detection_id):
    det = db.session.get(Detection, detection_id) or abort(404)
    new = request.form.get("status", "")
    if new not in DETECTION_STATUSES:
        abort(400)
    old, det.status = det.status, new
    audit.record("detection.status_changed", details={
        "detection_id": det.detection_id, "from": old, "to": new, "username": current_user.username})
    db.session.commit()
    flash(f"Detection #{det.detection_id} marked {new.replace('_', ' ')}.", "success")
    return redirect(url_for("dashboard.alert_detail", detection_id=detection_id))


@bp.route("/alerts/bulk-status", methods=["POST"])
@login_required
def bulk_status():
    new = request.form.get("status", "")
    ids = [int(i) for i in request.form.getlist("ids") if i.isdigit()]
    if new not in DETECTION_STATUSES or not ids:
        flash("Select at least one alert and a status.", "warning")
        return redirect(url_for("dashboard.alerts"))
    n = Detection.query.filter(Detection.detection_id.in_(ids)).update(
        {Detection.status: new}, synchronize_session=False)
    audit.record("detection.bulk_status_changed",
                 details={"ids": ids, "to": new, "username": current_user.username})
    db.session.commit()
    flash(f"{n} alert(s) marked {new.replace('_', ' ')}.", "success")
    return redirect(url_for("dashboard.alerts"))
