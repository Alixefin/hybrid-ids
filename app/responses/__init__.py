"""Response actions log, manual release / approval, on-demand expiry job."""
from flask import Blueprint

bp = Blueprint("responses", __name__, url_prefix="/responses")

from app.responses import routes  # noqa: E402,F401
