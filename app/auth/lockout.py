"""Account lock-out derived from audit_log.

A user is locked when there are >= MAX_FAILED_LOGINS "auth.login_failed" events since
their most recent "auth.login_success" / "auth.unlocked" / "auth.password_reset" event.
Deriving the state from the audit trail keeps the user table exactly as specified and
makes every lock / unlock auditable.
"""
from __future__ import annotations

from flask import current_app
from sqlalchemy import func

from app.models import AuditLog, db

RESET_EVENTS = ("auth.login_success", "auth.unlocked", "auth.password_reset")


def _event_like(name):
    return (AuditLog.event == name) | AuditLog.event.like(name + " %")


def failed_attempts(user_id: int) -> int:
    reset_filter = _event_like(RESET_EVENTS[0])
    for name in RESET_EVENTS[1:]:
        reset_filter = reset_filter | _event_like(name)
    last_reset = (db.session.query(func.max(AuditLog.log_id))
                  .filter(AuditLog.user_id == user_id, reset_filter).scalar()) or 0
    return (AuditLog.query
            .filter(AuditLog.user_id == user_id, _event_like("auth.login_failed"),
                    AuditLog.log_id > last_reset)
            .count())


def is_locked(user_id: int) -> bool:
    return failed_attempts(user_id) >= current_app.config["MAX_FAILED_LOGINS"]
