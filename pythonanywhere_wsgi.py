"""WSGI entry point for PythonAnywhere ("Manual configuration" web app).

Copy the contents of this file into the WSGI configuration file that PythonAnywhere
creates for you (Web tab -> "WSGI configuration file", e.g.
/var/www/<username>_pythonanywhere_com_wsgi.py) and replace <username> below.
SECRET_KEY: leave it unset and the app generates a random key on first start and keeps it
in instance/secret_key (git-ignored). Set os.environ["SECRET_KEY"] here only if you want
to manage it yourself.
"""
import os
import sys

PROJECT_HOME = "/home/<username>/hybrid_ids"  # <-- change <username>

if PROJECT_HOME not in sys.path:
    sys.path.insert(0, PROJECT_HOME)
os.chdir(PROJECT_HOME)

os.environ.setdefault("ENFORCER_MODE", "simulate")     # no root / netfilter on PythonAnywhere
os.environ.setdefault("SCHEDULER_ENABLED", "0")        # no background threads on the free tier
os.environ.setdefault("SESSION_COOKIE_SECURE", "1")    # enable "Force HTTPS" on the Web tab
# SQLite lives on persistent storage inside the project (never /tmp):
os.environ.setdefault("DATABASE_URL", f"sqlite:///{PROJECT_HOME}/instance/hybrid_ids.db")

from app import create_app  # noqa: E402

application = create_app()
