"""Authentication: login / logout / password change, 3-strike account lock-out."""
from flask import Blueprint

bp = Blueprint("auth", __name__)

from app.auth import routes  # noqa: E402,F401
