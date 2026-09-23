from flask import current_app, flash, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required, login_user, logout_user
from werkzeug.security import check_password_hash, generate_password_hash

from app import audit
from app.auth import bp
from app.auth.lockout import failed_attempts, is_locked
from app.models import User, db
from app.security import safe_next

# Compared against when the username does not exist, so response time does not reveal
# which usernames are valid.
_DUMMY_HASH = generate_password_hash("not-a-real-password")


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.index"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()
        limit = current_app.config["MAX_FAILED_LOGINS"]
        if user is None:
            check_password_hash(_DUMMY_HASH, password)
            audit.record("auth.login_failed", user_id=False,
                         details={"username": username[:64], "reason": "unknown user",
                                  "ip": request.remote_addr}, commit=True)
            flash("Invalid username or password.", "danger")
            return render_template("auth/login.html"), 401
        if is_locked(user.user_id):
            audit.record("auth.login_blocked", user_id=user.user_id,
                         details={"username": username, "ip": request.remote_addr}, commit=True)
            flash("This account is locked after too many failed logins. Ask an administrator "
                  "to unlock it.", "danger")
            return render_template("auth/login.html"), 403
        if not user.check_password(password):
            audit.record("auth.login_failed", user_id=user.user_id,
                         details={"username": username, "ip": request.remote_addr}, commit=True)
            left = limit - failed_attempts(user.user_id)
            if left <= 0:
                audit.record("auth.locked", user_id=user.user_id,
                             details={"username": username, "after_failures": limit}, commit=True)
                flash("Too many failed logins - the account is now locked.", "danger")
                return render_template("auth/login.html"), 403
            flash(f"Invalid username or password. {left} attempt(s) left before lock-out.", "danger")
            return render_template("auth/login.html"), 401
        session.clear()  # prevent session fixation
        login_user(user)
        audit.record("auth.login_success", user_id=user.user_id,
                     details={"username": username, "ip": request.remote_addr}, commit=True)
        return redirect(safe_next(request.args.get("next")))
    return render_template("auth/login.html")


@bp.route("/logout", methods=["POST"])
@login_required
def logout():
    audit.record("auth.logout", details={"username": current_user.username}, commit=True)
    logout_user()
    session.clear()
    flash("Signed out.", "info")
    return redirect(url_for("auth.login"))


@bp.route("/account/password", methods=["GET", "POST"])
@login_required
def change_password():
    if request.method == "POST":
        current = request.form.get("current_password", "")
        new = request.form.get("new_password", "")
        confirm = request.form.get("confirm_password", "")
        if not current_user.check_password(current):
            flash("Current password is wrong.", "danger")
        elif len(new) < 10:
            flash("New password must be at least 10 characters.", "danger")
        elif new != confirm:
            flash("New passwords do not match.", "danger")
        else:
            current_user.set_password(new)
            audit.record("auth.password_changed", details={"username": current_user.username})
            db.session.commit()
            flash("Password changed.", "success")
            return redirect(url_for("dashboard.index"))
    return render_template("auth/change_password.html")
