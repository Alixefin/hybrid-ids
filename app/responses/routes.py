from flask import abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from app import services
from app.models import ACTION_STATUSES, Detection, ResponseAction, db
from app.responses import bp
from app.security import admin_required, safe_next
from engine.policy import ACTION_TYPES

PER_PAGE = 50


@bp.route("/")
@login_required
def index():
    f = {k: request.args.get(k, "").strip() for k in ("action_type", "status", "mode", "ip")}
    q = ResponseAction.query
    if f["action_type"] in ACTION_TYPES:
        q = q.filter(ResponseAction.action_type == f["action_type"])
    if f["status"] in ACTION_STATUSES:
        q = q.filter(ResponseAction.status == f["status"])
    if f["mode"] in ("simulate", "enforce"):
        q = q.filter(ResponseAction.mode == f["mode"])
    if f["ip"]:
        q = q.filter(ResponseAction.target_ip.like(f"%{f['ip']}%"))
    page = max(1, request.args.get("page", 1, type=int))
    pagination = q.order_by(ResponseAction.action_id.desc()).paginate(
        page=page, per_page=PER_PAGE, error_out=False)
    args = {k: v for k, v in f.items() if v}
    return render_template("responses/index.html", pagination=pagination, f=f, args=args,
                           statuses=ACTION_STATUSES, action_types=ACTION_TYPES)


@bp.route("/<int:action_id>/release", methods=["POST"])
@admin_required
def release(action_id):
    action = db.session.get(ResponseAction, action_id) or abort(404)
    try:
        services.release_action(action, current_user.user_id)
        flash(f"Action #{action_id} released ({action.status}).", "success")
    except ValueError as exc:
        flash(str(exc), "warning")
    return redirect(safe_next(request.form.get("next"), "responses.index"))


@bp.route("/expire-now", methods=["POST"])
@admin_required
def expire_now():
    n = services.expire_actions(user_id=current_user.user_id)
    flash(f"Expiry job ran: {n} action(s) expired and released.", "success")
    return redirect(safe_next(request.form.get("next"), "responses.index"))


@bp.route("/manual/<int:detection_id>", methods=["POST"])
@admin_required
def manual(detection_id):
    det = db.session.get(Detection, detection_id) or abort(404)
    action_type = request.form.get("action_type", "")
    target = request.form.get("target_ip", "").strip()
    try:
        duration = int(request.form.get("duration_min", "0") or 0)
        action = services.manual_action(det, action_type, target, duration, current_user.user_id)
        level = "success" if action.status != "failed" else "warning"
        flash(f"Manual action #{action.action_id} recorded with status '{action.status}'.", level)
    except ValueError as exc:
        db.session.rollback()
        flash(f"Not executed: {exc}", "danger")
    return redirect(url_for("dashboard.alert_detail", detection_id=detection_id))
