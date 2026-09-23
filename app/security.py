"""CSRF protection, security headers and role-based access control."""
from __future__ import annotations

import hmac
import secrets
from functools import wraps
from urllib.parse import urlparse

from flask import abort, current_app, request, session, url_for
from flask_login import current_user

CSRF_FIELD = "csrf_token"


def csrf_token() -> str:
    token = session.get("_csrf")
    if not token:
        token = session["_csrf"] = secrets.token_urlsafe(32)
    return token


def _check_csrf():
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return
    if not current_app.config.get("CSRF_ENABLED", True):
        return
    sent = request.form.get(CSRF_FIELD) or request.headers.get("X-CSRF-Token", "")
    expected = session.get("_csrf", "")
    if not expected or not hmac.compare_digest(str(sent), str(expected)):
        abort(400, "CSRF token missing or invalid - reload the page and try again.")


def _headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    resp.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "font-src 'self' https://cdn.jsdelivr.net; img-src 'self' data:; frame-ancestors 'none'",
    )
    return resp


def init_app(app):
    app.config.setdefault("CSRF_ENABLED", not app.config.get("TESTING", False))
    app.before_request(_check_csrf)
    app.after_request(_headers)
    app.jinja_env.globals["csrf_token"] = csrf_token


def role_required(*roles):
    """Allow only logged-in users whose role is in ``roles``."""
    def decorator(view):
        @wraps(view)
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated:
                return current_app.login_manager.unauthorized()
            if current_user.role not in roles:
                abort(403)
            return view(*args, **kwargs)
        return wrapper
    return decorator


admin_required = role_required("admin")


def safe_next(target: str | None, default_endpoint: str = "dashboard.index") -> str:
    """Only allow same-site relative redirect targets (prevents open redirects)."""
    if (target and target.startswith("/") and not target.startswith("//")
            and "\\" not in target and not urlparse(target).netloc):
        return target
    return url_for(default_endpoint)
