"""Flask integration: auth + lock-out, RBAC, CSRF, and the synchronous detection flow."""
import io
from datetime import timedelta

import pytest

from app import services
from app.models import AuditLog, Detection, FlowRecord, ResponseAction, Whitelist, db
from engine.ingest import utcnow
from tests.conftest import make_user


def login(client, username="admin", password="correct-horse-1"):
    return client.post("/login", data={"username": username, "password": password})


def test_login_and_lockout_after_three_failures(app, client):
    make_user()
    assert login(client, password="nope").status_code == 401
    assert login(client, password="nope").status_code == 401
    r = login(client, password="nope")
    assert r.status_code == 403 and b"locked" in r.data
    # even the right password is refused now
    assert login(client).status_code == 403
    events = [e.event.split(" ")[0] for e in AuditLog.query.order_by(AuditLog.log_id)]
    assert events.count("auth.login_failed") == 3 and "auth.locked" in events and "auth.login_blocked" in events


def test_successful_login_resets_failure_count(app, client):
    make_user()
    login(client, password="nope")
    login(client, password="nope")
    assert login(client).status_code == 302
    client.post("/logout")
    login(client, password="nope")
    assert login(client).status_code == 302  # only 1 failure since the last success


def test_admin_unlock(app, client):
    u = make_user("bob", role="analyst")
    for _ in range(3):
        login(client, "bob", "nope")
    admin = app.test_client()
    make_user("root", role="admin")
    login(admin, "root")
    admin.post(f"/admin/users/{u.user_id}", data={"op": "unlock"})
    assert login(client, "bob").status_code == 302


def test_unknown_user_is_audited_without_user_id(app, client):
    assert login(client, "ghost", "x").status_code == 401
    e = AuditLog.query.one()
    assert e.user_id is None and "unknown user" in e.event


def test_passwords_are_hashed(app):
    u = make_user()
    assert u.password_hash != "correct-horse-1" and u.password_hash.startswith(("scrypt:", "pbkdf2:"))
    assert u.check_password("correct-horse-1") and not u.check_password("wrong")


def test_login_required_and_rbac(app, client):
    assert client.get("/").status_code == 302  # -> login
    make_user("ana", role="analyst")
    login(client, "ana")
    assert client.get("/").status_code == 200
    assert client.get("/alerts").status_code == 200
    assert client.get("/admin/evaluate").status_code == 200       # analysts may run offline evaluation
    for url in ("/admin/models", "/admin/policies", "/admin/users", "/admin/audit"):
        assert client.get(url).status_code == 403, url
    assert client.post("/responses/expire-now").status_code == 403
    assert client.post("/admin/whitelist", data={"ip_or_cidr": "1.2.3.4"}).status_code == 403


def test_open_redirect_blocked(app, client):
    make_user()
    r = client.post("/login?next=https://evil.example/", data={"username": "admin", "password": "correct-horse-1"})
    assert r.headers["Location"] == "/"


def test_csrf_enforced_when_enabled(app, client):
    app.config["CSRF_ENABLED"] = True
    make_user()
    assert login(client).status_code == 400          # no token
    client.get("/login")
    with client.session_transaction() as s:
        token = s["_csrf"]
    r = client.post("/login", data={"username": "admin", "password": "correct-horse-1", "csrf_token": token})
    assert r.status_code == 302


def test_security_headers(app, client):
    r = client.get("/login")
    assert r.headers["X-Frame-Options"] == "DENY"
    assert "default-src 'self'" in r.headers["Content-Security-Policy"]


def _csv_bytes(frame):
    from training.generate_synthetic_flows import to_raw_csv_frame
    buf = io.StringIO()
    to_raw_csv_frame(frame).to_csv(buf, index=False)
    return buf.getvalue().encode()


@pytest.fixture
def upload_bytes():
    from training.generate_synthetic_flows import generate
    return _csv_bytes(generate(1500, seed=11, min_per_class=15))


def test_offline_evaluation_upload_end_to_end(app, client, upload_bytes):
    services.register_models_from_store()
    make_user()
    login(client)
    r = client.post("/admin/evaluate", data={"file": (io.BytesIO(upload_bytes), "flows.csv"), "persist": "on"},
                    content_type="multipart/form-data")
    assert r.status_code == 302 and "/admin/evaluate/results/" in r.headers["Location"]
    page = client.get(r.headers["Location"])
    assert page.status_code == 200 and b"Detection results" in page.data and b"synthetic" in page.data

    run = AuditLog.query.filter(AuditLog.event.like("detection_run %")).one()
    assert '"total": 1515' in run.event                     # 1500 flows + 1% duplicates
    stored = Detection.query.count()
    assert 0 < stored < 1515                                  # benign below theta is not stored
    assert FlowRecord.query.count() == stored
    flagged = Detection.query.filter(Detection.predicted_class != "Benign").all()
    assert flagged and all(d.status == "new" for d in flagged)
    assert all(d.status == "resolved" for d in Detection.query.filter_by(predicted_class="Benign"))
    actions = ResponseAction.query.all()
    assert actions and all(a.mode == "simulate" for a in actions)
    assert {a.status for a in actions} <= {"simulated", "active", "failed"}
    # every action is audited
    assert AuditLog.query.filter(AuditLog.event.like("response.%")).count() == len(actions)
    # the dashboard pages render with data
    for url in ("/", "/alerts", f"/alerts/{flagged[0].detection_id}", "/responses/", "/admin/models"):
        assert client.get(url).status_code == 200, url


def test_dry_run_stores_nothing(app, client, upload_bytes, tmp_path):
    services.register_models_from_store()
    p = tmp_path / "flows.csv"
    p.write_bytes(upload_bytes)
    summary = services.run_detection(p, persist=False, user_id=False)
    assert summary["counts"]["total"] == 1515 and summary["ground_truth"]["hybrid_binary"]["recall"] > 0
    assert Detection.query.count() == 0 and ResponseAction.query.count() == 0


def test_whitelisted_attacker_is_never_blocked(app, upload_bytes, tmp_path):
    services.register_models_from_store()
    db.session.add(Whitelist(ip_or_cidr="172.16.0.0/12", reason="test"))
    db.session.commit()
    p = tmp_path / "flows.csv"
    p.write_bytes(upload_bytes)
    services.run_detection(p, user_id=False)
    on_attacker = ResponseAction.query.filter_by(target_ip="172.16.0.1").all()
    assert on_attacker
    for a in on_attacker:
        if a.action_type != "alert":
            assert a.status == "failed" and "whitelisted" in a.outcome


def test_expiry_job_releases_due_actions(app, upload_bytes, tmp_path):
    services.register_models_from_store()
    p = tmp_path / "flows.csv"
    p.write_bytes(upload_bytes)
    services.run_detection(p, user_id=False)
    timed = ResponseAction.query.filter(ResponseAction.expires_at.isnot(None)).all()
    assert timed
    for a in timed:
        a.expires_at = utcnow() - timedelta(minutes=1)
    db.session.commit()
    assert services.expire_actions() == len(timed)
    assert all(db.session.get(ResponseAction, a.action_id).status == "expired" for a in timed)
    assert AuditLog.query.filter(AuditLog.event.like("job.expire_actions %")).count() == 1


def test_repeat_offence_doubles_next_block(app, upload_bytes, tmp_path):
    services.register_models_from_store()
    p = tmp_path / "flows.csv"
    p.write_bytes(upload_bytes)
    services.run_detection(p, user_id=False)
    first = ResponseAction.query.filter_by(target_ip="172.16.0.1", action_type="temp_block").first()
    assert first is not None
    first.status = "expired"  # the block ran out; the attacker comes back
    db.session.commit()
    services.run_detection(p, user_id=False)
    second = (ResponseAction.query.filter_by(target_ip="172.16.0.1", action_type="temp_block")
              .order_by(ResponseAction.action_id.desc()).first())
    assert second.action_id != first.action_id
    base = (second.expires_at - second.executed_at).total_seconds() / 60
    assert base == pytest.approx(2 * (first.expires_at - first.executed_at).total_seconds() / 60, abs=0.1)


def test_manual_action_records_approver(app, client, upload_bytes, tmp_path):
    services.register_models_from_store()
    admin = make_user()
    p = tmp_path / "flows.csv"
    p.write_bytes(upload_bytes)
    services.run_detection(p, user_id=False)
    det = Detection.query.filter(Detection.predicted_class != "Benign").first()
    login(client)
    client.post(f"/responses/manual/{det.detection_id}",
                data={"action_type": "temp_block", "target_ip": "203.0.113.9", "duration_min": "30"})
    a = ResponseAction.query.filter_by(target_ip="203.0.113.9").one()
    assert a.approved_by == admin.user_id and a.status == "simulated"
    client.post(f"/responses/manual/{det.detection_id}",
                data={"action_type": "temp_block", "target_ip": "1.2.3.4; reboot", "duration_min": "30"})
    assert ResponseAction.query.filter(ResponseAction.target_ip.like("%reboot%")).count() == 0


def test_cli_init_db_and_create_admin(app):
    runner = app.test_cli_runner()
    r = runner.invoke(args=["init-db"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(args=["create-admin", "--username", "boss", "--email", "", "--password", "long-enough-pw"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(args=["expire-actions"])
    assert "expired" in r.output
