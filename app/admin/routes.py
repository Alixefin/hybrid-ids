import json
import re
import uuid
from pathlib import Path

from flask import abort, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from sqlalchemy.exc import IntegrityError
from werkzeug.utils import secure_filename

from app import audit, services, settings
from app.admin import bp
from app.audit import parse_details
from app.auth.lockout import failed_attempts, is_locked
from app.models import (
    ROLES, AuditLog, ModelVersion, ResponseAction, ResponsePolicy, User, Whitelist, db,
)
from app.security import admin_required, role_required
from engine.enforcer import IptablesEnforcer
from engine.policy import ACTION_LABELS, ACTION_TYPES, SEVERITIES, normalise_whitelist_entry
from engine.schema import THREAT_CATEGORIES, UNKNOWN_ANOMALY

ALLOWED_UPLOAD = re.compile(r"^[\w.\- ]+\.csv$", re.IGNORECASE)


# ================================================================ models / thresholds
@bp.route("/models")
@admin_required
def models():
    versions = ModelVersion.query.order_by(ModelVersion.trained_at.desc(), ModelVersion.model_id.desc()).all()
    runs = {}
    for mv in versions:
        runs.setdefault(mv.run_id, {})[mv.model_type] = mv
    report_path = Path(current_app.config["BASE_DIR"]) / "reports" / "latest_metrics.json"
    return render_template("admin/models.html", runs=runs, runtime=settings.all_settings(),
                           iptables_available=IptablesEnforcer.available(),
                           has_report=report_path.exists())


@bp.route("/models/register", methods=["POST"])
@admin_required
def models_register():
    added = services.register_models_from_store()
    flash(f"{len(added)} new model file(s) registered." if added else "No new model files found.", "info")
    return redirect(url_for("admin.models"))


@bp.route("/models/activate", methods=["POST"])
@admin_required
def models_activate():
    try:
        services.activate_run(request.form.get("run_id", ""), user_id=current_user.user_id)
        flash("Models activated.", "success")
    except ValueError as exc:
        flash(str(exc), "danger")
    return redirect(url_for("admin.models"))


@bp.route("/models/settings", methods=["POST"])
@admin_required
def models_settings():
    values = {k: request.form[k] for k in ("theta_percentile", "tau", "enforcer_mode") if k in request.form}
    if values.get("enforcer_mode") == "enforce" and not IptablesEnforcer.available():
        flash("Enforce mode needs iptables/conntrack and root on a Linux host - it cannot work "
              "here (e.g. on PythonAnywhere). Staying in simulate mode.", "danger")
        values.pop("enforcer_mode")
    try:
        before = settings.all_settings()
        after = settings.update(**values)
        services.clear_detector_cache()
        audit.record("settings.changed", details={"before": before, "after": after})
        db.session.commit()
        flash("Detection settings saved.", "success")
    except ValueError as exc:
        flash(str(exc), "danger")
    return redirect(url_for("admin.models"))


@bp.route("/report")
@admin_required
def report():
    path = Path(current_app.config["BASE_DIR"]) / "reports" / "latest_metrics.json"
    if not path.exists():
        flash("No evaluation report yet - run: python -m training.evaluate", "warning")
        return redirect(url_for("admin.models"))
    return render_template("admin/report.html", r=json.loads(path.read_text()))


# ================================================================ policy / whitelist
@bp.route("/policies")
@admin_required
def policies():
    rows = ResponsePolicy.query.order_by(ResponsePolicy.threat_class,
                                         ResponsePolicy.min_confidence.desc(),
                                         ResponsePolicy.policy_id).all()
    return render_template("admin/policies.html", rows=rows, whitelist=Whitelist.query.order_by(
        Whitelist.entry_id).all(), classes=THREAT_CATEGORIES[1:] + [UNKNOWN_ANOMALY],
        action_types=ACTION_TYPES, severities=SEVERITIES)


def _policy_from_form(p: ResponsePolicy):
    cls = request.form.get("threat_class", p.threat_class)
    sev = request.form.get("severity", p.severity)
    act = request.form.get("action_type", p.action_type)
    if cls not in THREAT_CATEGORIES + [UNKNOWN_ANOMALY] or sev not in SEVERITIES or act not in ACTION_TYPES:
        raise ValueError("invalid class, severity or action")
    conf = float(request.form.get("min_confidence", p.min_confidence))
    dur = int(request.form.get("duration_min", p.duration_min))
    if not 0.0 <= conf <= 1.0 or not 0 <= dur <= 24 * 60:
        raise ValueError("confidence must be 0-1 and duration 0-1440 minutes")
    if cls == UNKNOWN_ANOMALY and act != "alert":
        raise ValueError("Unknown anomalies may only raise alerts (no auto-block)")
    p.threat_class, p.severity, p.action_type, p.min_confidence, p.duration_min = cls, sev, act, conf, dur
    p.enabled = request.form.get("enabled") == "on"


@bp.route("/policies/<int:policy_id>", methods=["POST"])
@admin_required
def policy_update(policy_id):
    p = db.session.get(ResponsePolicy, policy_id) or abort(404)
    if request.form.get("delete"):
        audit.record("policy.deleted", details={"policy_id": policy_id, "class": p.threat_class,
                                                "action": p.action_type})
        db.session.delete(p)
        db.session.commit()
        flash(f"Policy #{policy_id} deleted.", "info")
        return redirect(url_for("admin.policies"))
    try:
        _policy_from_form(p)
        audit.record("policy.updated", details={"policy_id": policy_id, "class": p.threat_class,
                                                "min_confidence": p.min_confidence, "action": p.action_type,
                                                "duration_min": p.duration_min, "enabled": p.enabled})
        db.session.commit()
        flash(f"Policy #{policy_id} saved.", "success")
    except ValueError as exc:
        db.session.rollback()
        flash(f"Not saved: {exc}", "danger")
    return redirect(url_for("admin.policies"))


@bp.route("/policies/new", methods=["POST"])
@admin_required
def policy_create():
    p = ResponsePolicy(threat_class="", severity="Medium", min_confidence=0.0, action_type="alert",
                       duration_min=0, enabled=True)
    try:
        _policy_from_form(p)
        db.session.add(p)
        db.session.flush()
        audit.record("policy.created", details={"policy_id": p.policy_id, "class": p.threat_class,
                                                "action": p.action_type})
        db.session.commit()
        flash("Policy rule added.", "success")
    except ValueError as exc:
        db.session.rollback()
        flash(f"Not added: {exc}", "danger")
    return redirect(url_for("admin.policies"))


@bp.route("/policies/reset", methods=["POST"])
@admin_required
def policy_reset():
    ResponseAction.query.filter(ResponseAction.policy_id.isnot(None)).update(
        {ResponseAction.policy_id: None}, synchronize_session=False)
    n = services.seed_default_policies(replace=True)
    audit.record("policy.reset_to_defaults", details={"rules": n})
    db.session.commit()
    flash(f"Policy table reset to the {n} default rules.", "info")
    return redirect(url_for("admin.policies"))


@bp.route("/whitelist", methods=["POST"])
@admin_required
def whitelist_add():
    try:
        entry = normalise_whitelist_entry(request.form.get("ip_or_cidr", ""))
        reason = request.form.get("reason", "").strip()[:255]
        db.session.add(Whitelist(ip_or_cidr=entry, reason=reason))
        audit.record("whitelist.added", details={"entry": entry, "reason": reason})
        db.session.commit()
        flash(f"{entry} added to the whitelist.", "success")
    except ValueError as exc:
        flash(f"Invalid IP / CIDR: {exc}", "danger")
    except IntegrityError:
        db.session.rollback()
        flash("That entry is already whitelisted.", "warning")
    return redirect(url_for("admin.policies"))


@bp.route("/whitelist/<int:entry_id>/delete", methods=["POST"])
@admin_required
def whitelist_delete(entry_id):
    w = db.session.get(Whitelist, entry_id) or abort(404)
    audit.record("whitelist.removed", details={"entry": w.ip_or_cidr})
    db.session.delete(w)
    db.session.commit()
    flash(f"{w.ip_or_cidr} removed from the whitelist.", "info")
    return redirect(url_for("admin.policies"))


# ================================================================ offline evaluation
def _stored_files():
    out = []
    for folder, label in ((Path(current_app.config["UPLOAD_DIR"]), "upload"),
                          (Path(current_app.config["BASE_DIR"]) / "data" / "samples", "sample")):
        if folder.exists():
            for p in sorted(folder.glob("*.csv")):
                out.append({"name": p.name, "kind": label, "size_kb": p.stat().st_size // 1024})
    return out


def _resolve_stored(kind: str, name: str) -> Path:
    folder = {"upload": Path(current_app.config["UPLOAD_DIR"]),
              "sample": Path(current_app.config["BASE_DIR"]) / "data" / "samples"}.get(kind)
    if folder is None or name != secure_filename(name) or not ALLOWED_UPLOAD.match(name):
        abort(400)
    path = (folder / name).resolve()
    if path.parent != folder.resolve() or not path.exists():
        abort(404)
    return path


def _recent_runs(limit=10):
    rows = (AuditLog.query.filter(AuditLog.event.like("detection_run %"))
            .order_by(AuditLog.log_id.desc()).limit(limit).all())
    return [(r, parse_details(r.event)[1]) for r in rows]


@bp.route("/evaluate", methods=["GET", "POST"])
@role_required("analyst", "admin")
def evaluate():
    if request.method == "POST":
        f = request.files.get("file")
        if not f or not f.filename:
            flash("Choose a CICFlowMeter CSV file to upload.", "warning")
            return redirect(url_for("admin.evaluate"))
        name = secure_filename(f.filename)
        if not ALLOWED_UPLOAD.match(name):
            flash("Only .csv files are accepted.", "danger")
            return redirect(url_for("admin.evaluate"))
        dest = Path(current_app.config["UPLOAD_DIR"]) / f"{uuid.uuid4().hex[:8]}_{name}"
        f.save(dest)
        audit.record("upload.saved", details={"file": dest.name, "bytes": dest.stat().st_size})
        db.session.commit()
        return _run_and_redirect(dest, request.form.get("persist") == "on")
    return render_template("admin/evaluate.html", files=_stored_files(), runs=_recent_runs(),
                           max_rows=current_app.config["MAX_UPLOAD_ROWS"],
                           max_mb=current_app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024))


@bp.route("/evaluate/run-file", methods=["POST"])
@admin_required
def evaluate_stored():
    """'Run detection on this file' - re-score a file already on the server."""
    path = _resolve_stored(request.form.get("kind", ""), request.form.get("name", ""))
    return _run_and_redirect(path, request.form.get("persist") == "on")


@bp.route("/evaluate/delete-file", methods=["POST"])
@admin_required
def evaluate_delete():
    path = _resolve_stored("upload", request.form.get("name", ""))
    path.unlink()
    audit.record("upload.deleted", details={"file": path.name})
    db.session.commit()
    flash(f"{path.name} deleted.", "info")
    return redirect(url_for("admin.evaluate"))


def _run_and_redirect(path: Path, persist: bool):
    try:
        summary = services.run_detection(path, source_name=path.name, persist=persist,
                                         user_id=current_user.user_id)
    except services.NoActiveModels as exc:
        db.session.rollback()
        flash(str(exc), "danger")
        return redirect(url_for("admin.evaluate"))
    except ValueError as exc:
        db.session.rollback()
        flash(f"Could not score {path.name}: {exc}", "danger")
        return redirect(url_for("admin.evaluate"))
    return redirect(url_for("admin.evaluate_result", run_id=summary["run_id"]))


@bp.route("/evaluate/results/<run_id>")
@login_required
def evaluate_result(run_id):
    s = services.load_run_summary(run_id) or abort(404)
    return render_template("admin/evaluate_result.html", s=s)


# ============================================================================= users
@bp.route("/users")
@admin_required
def users():
    rows = User.query.order_by(User.user_id).all()
    status = {u.user_id: {"locked": is_locked(u.user_id), "failed": failed_attempts(u.user_id)} for u in rows}
    return render_template("admin/users.html", rows=rows, status=status, roles=ROLES)


@bp.route("/users/new", methods=["POST"])
@admin_required
def user_create():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    role = request.form.get("role", "analyst")
    email = request.form.get("email", "").strip() or None
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,64}", username):
        flash("Username: 3-64 letters, digits, '.', '_' or '-'.", "danger")
    elif len(password) < 10:
        flash("Password must be at least 10 characters.", "danger")
    elif role not in ROLES:
        flash("Invalid role.", "danger")
    elif User.query.filter_by(username=username).first():
        flash("That username exists.", "danger")
    else:
        u = User(username=username, role=role, email=email)
        u.set_password(password)
        db.session.add(u)
        db.session.flush()
        audit.record("user.created", details={"username": username, "role": role,
                                              "new_user_id": u.user_id})
        db.session.commit()
        flash(f"User '{username}' created.", "success")
    return redirect(url_for("admin.users"))


@bp.route("/users/<int:user_id>", methods=["POST"])
@admin_required
def user_update(user_id):
    u = db.session.get(User, user_id) or abort(404)
    op = request.form.get("op")
    if op == "role":
        role = request.form.get("role")
        if role not in ROLES:
            abort(400)
        if u.user_id == current_user.user_id and role != "admin":
            flash("You cannot remove your own admin role.", "warning")
            return redirect(url_for("admin.users"))
        u.role = role
        audit.record("user.role_changed", details={"username": u.username, "role": role})
    elif op == "unlock":
        audit.record("auth.unlocked", user_id=u.user_id,
                     details={"username": u.username, "by": current_user.username})
    elif op == "password":
        pw = request.form.get("password", "")
        if len(pw) < 10:
            flash("Password must be at least 10 characters.", "danger")
            return redirect(url_for("admin.users"))
        u.set_password(pw)
        audit.record("auth.password_reset", user_id=u.user_id,
                     details={"username": u.username, "by": current_user.username})
    elif op == "delete":
        if u.user_id == current_user.user_id:
            flash("You cannot delete your own account.", "warning")
            return redirect(url_for("admin.users"))
        AuditLog.query.filter_by(user_id=u.user_id).update({AuditLog.user_id: None})
        ResponseAction.query.filter_by(approved_by=u.user_id).update({ResponseAction.approved_by: None})
        audit.record("user.deleted", details={"username": u.username, "deleted_user_id": u.user_id})
        db.session.delete(u)
    else:
        abort(400)
    db.session.commit()
    flash("User updated.", "success")
    return redirect(url_for("admin.users"))


# ============================================================================= audit
@bp.route("/audit")
@admin_required
def audit_log():
    q = AuditLog.query
    term = request.args.get("q", "").strip()
    if term:
        q = q.filter(AuditLog.event.like(f"%{term}%"))
    page = max(1, request.args.get("page", 1, type=int))
    pagination = q.order_by(AuditLog.log_id.desc()).paginate(page=page, per_page=100, error_out=False)
    return render_template("admin/audit.html", pagination=pagination, term=term,
                           action_labels=ACTION_LABELS)
