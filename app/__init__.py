"""Flask application factory."""
from __future__ import annotations

import logging
import os

from flask import Flask, render_template
from flask_login import LoginManager
from sqlalchemy import event
from sqlalchemy.engine import Engine

from app.models import db, User
from config import Config

login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message_category = "warning"


@event.listens_for(Engine, "connect")
def _sqlite_pragmas(dbapi_connection, _record):
    # SQLite ignores foreign keys unless asked. (No WAL: PythonAnywhere's file storage
    # does not reliably support its shared-memory file, and there is only one worker.)
    if dbapi_connection.__class__.__module__.startswith("sqlite3"):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


@login_manager.user_loader
def _load_user(user_id: str):
    return db.session.get(User, int(user_id))


def create_app(config_object=Config) -> Flask:
    app = Flask(__name__, instance_path=str(config_object.INSTANCE_DIR), instance_relative_config=False)
    app.config.from_object(config_object)
    for key in ("INSTANCE_DIR", "UPLOAD_DIR", "MODELS_DIR"):
        os.makedirs(app.config[key], exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    db.init_app(app)
    login_manager.init_app(app)

    from app import security
    security.init_app(app)

    from app.admin import bp as admin_bp
    from app.auth import bp as auth_bp
    from app.dashboard import bp as dashboard_bp
    from app.responses import bp as responses_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(responses_bp)
    app.register_blueprint(admin_bp)

    from app import cli, context
    cli.register(app)
    context.register(app)

    @app.errorhandler(403)
    def _forbidden(_e):
        return render_template("error.html", code=403, message="You do not have permission to do that."), 403

    @app.errorhandler(404)
    def _not_found(_e):
        return render_template("error.html", code=404, message="Page not found."), 404

    @app.errorhandler(413)
    def _too_large(_e):
        mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return render_template("error.html", code=413, message=f"Upload too large (limit {mb} MB)."), 413

    if app.config.get("SCHEDULER_ENABLED") and not app.config.get("TESTING"):
        from app.scheduler import start_scheduler
        start_scheduler(app)

    return app
