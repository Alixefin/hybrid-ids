"""Flask CLI commands:  flask --app run.py <command>

    init-db            create tables, seed the default policy table and whitelist,
                       register trained models from models_store/
    create-admin       create (or reset) an admin account
    register-models    import new models_store/run_*.json manifests
    run-detection      score a CSV synchronously (same code path as the upload page)
    expire-actions     release expired response actions (use as the PythonAnywhere
                       daily scheduled task)
"""
from __future__ import annotations

import click
from flask import current_app

from app import audit, services
from app.models import ROLE_ADMIN, User, Whitelist, db

DEFAULT_WHITELIST = [
    ("127.0.0.0/8", "Loopback"),
    ("192.168.10.3", "Domain controller / DNS server (CICIDS2017 testbed)"),
]


def register(app):
    @app.cli.command("init-db")
    @click.option("--drop", is_flag=True, help="drop all tables first (DESTROYS DATA)")
    def init_db(drop):
        """Create tables and seed defaults."""
        if drop:
            click.confirm("Drop every table and all data?", abort=True)
            db.drop_all()
        db.create_all()
        n = services.seed_default_policies()
        if not Whitelist.query.first():
            for ip, reason in DEFAULT_WHITELIST:
                db.session.add(Whitelist(ip_or_cidr=ip, reason=reason))
        audit.record("system.init_db", user_id=False, details={"policies_seeded": n})
        db.session.commit()
        added = services.register_models_from_store()
        click.echo(f"Database ready at {current_app.config['SQLALCHEMY_DATABASE_URI']}")
        click.echo(f"  {n} default policy rows seeded; {len(added)} model file(s) registered")

    @app.cli.command("create-admin")
    @click.option("--username", prompt=True)
    @click.option("--email", default="", prompt="Email (optional)", show_default=False)
    @click.password_option()
    def create_admin(username, email, password):
        """Create an admin user (or reset an existing user's password and make it admin)."""
        if len(password) < 10:
            raise click.BadParameter("use at least 10 characters", param_hint="password")
        user = User.query.filter_by(username=username).first()
        created = user is None
        if created:
            user = User(username=username, role=ROLE_ADMIN, email=email or None)
            db.session.add(user)
        user.role = ROLE_ADMIN
        user.set_password(password)
        db.session.flush()
        audit.record("user.created" if created else "auth.password_reset", user_id=False,
                     details={"username": username, "role": ROLE_ADMIN, "via": "cli"})
        audit.record("auth.unlocked", user_id=user.user_id, details={"username": username, "via": "cli"})
        db.session.commit()
        click.echo(f"Admin '{username}' {'created' if created else 'updated'}.")

    @app.cli.command("register-models")
    def register_models():
        added = services.register_models_from_store()
        click.echo(f"{len(added)} new model file(s) registered: {', '.join(added) or '-'}")

    @app.cli.command("run-detection")
    @click.argument("csv_path", type=click.Path(exists=True, dir_okay=False))
    @click.option("--dry-run", is_flag=True, help="score only - do not store or respond")
    def run_detection(csv_path, dry_run):
        s = services.run_detection(csv_path, persist=not dry_run, user_id=False)
        click.echo(f"{s['rows_scored']:,} flows scored in {s['elapsed_s']['total']} s -> {s['counts']}")
        click.echo(f"flagged by class: {s['flagged_by_class']}")
        click.echo(f"actions: {s['actions']}")
        if s["data_source"] == "synthetic":
            click.echo("WARNING: active models were trained on SYNTHETIC data.")

    @app.cli.command("expire-actions")
    def expire_actions():
        n = services.expire_actions()
        click.echo(f"{n} action(s) expired")
