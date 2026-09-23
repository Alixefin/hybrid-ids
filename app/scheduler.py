"""Optional APScheduler job for action expiry.

Only started when SCHEDULER_ENABLED=1 (self-hosted lab VM). PythonAnywhere's free tier
cannot keep a background thread alive, so there the same job runs from the admin
"Run expiry now" button, `flask expire-actions`, or a daily scheduled task.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("hybrid_ids.scheduler")
_scheduler = None


def start_scheduler(app):
    global _scheduler
    if _scheduler is not None:
        return _scheduler
    # Under the Werkzeug reloader only the child process should run the job.
    if app.debug and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        return None
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except ImportError:
        log.warning("APScheduler not installed - expiry job not scheduled")
        return None

    def job():
        with app.app_context():
            from app.services import expire_actions
            n = expire_actions()
            if n:
                log.info("expiry job released %d action(s)", n)

    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(job, "interval", minutes=app.config["EXPIRY_JOB_INTERVAL_MIN"],
                       id="expire_actions", max_instances=1, coalesce=True)
    _scheduler.start()
    log.info("APScheduler started: expiry job every %s min", app.config["EXPIRY_JOB_INTERVAL_MIN"])
    return _scheduler
