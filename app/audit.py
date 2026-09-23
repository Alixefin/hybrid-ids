"""audit_log helper - every automated or manual action goes through ``record``."""
from __future__ import annotations

import json

from flask_login import current_user

from app.models import AuditLog, db


def record(event: str, user_id=None, details: dict | None = None, commit: bool = False) -> AuditLog:
    """Add an audit entry. ``user_id=None`` means a system event, unless called inside a
    request by a logged-in user and ``user_id`` is not given explicitly (pass
    ``user_id=False`` to force a system event)."""
    if user_id is None:
        try:
            if current_user and current_user.is_authenticated:
                user_id = current_user.user_id
        except (RuntimeError, AttributeError):  # outside a request context
            user_id = None
    if user_id is False:
        user_id = None
    text = event if not details else f"{event} {json.dumps(details, default=str, sort_keys=True)}"
    entry = AuditLog(user_id=user_id, event=text)
    db.session.add(entry)
    if commit:
        db.session.commit()
    return entry


def parse_details(event_text: str) -> tuple[str, dict]:
    """Split 'name {json}' back into (name, dict)."""
    name, _, rest = event_text.partition(" ")
    if rest.startswith("{"):
        try:
            return name, json.loads(rest)
        except ValueError:
            pass
    return name, {"text": rest} if rest else {}
