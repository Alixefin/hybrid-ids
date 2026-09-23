"""SQLAlchemy models - the eight tables of the design, field-for-field.

Account lock-out (3 failed logins) is derived from audit_log events instead of extra
columns on ``user`` so the schema stays exactly as specified: see
``app.auth.lockout``.
"""
from __future__ import annotations

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

from engine.ingest import utcnow

db = SQLAlchemy()

ROLE_ANALYST = "analyst"
ROLE_ADMIN = "admin"
ROLES = (ROLE_ANALYST, ROLE_ADMIN)

DETECTION_STATUSES = ("new", "acknowledged", "resolved", "false_positive")
ACTION_STATUSES = ("active", "expired", "released", "failed", "simulated")
FLOW_SOURCES = ("live", "csv", "replay")
MODEL_TYPES = ("isolation_forest", "random_forest")


class User(UserMixin, db.Model):
    __tablename__ = "user"
    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(16), nullable=False, default=ROLE_ANALYST)
    email = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    __table_args__ = (db.CheckConstraint("role IN ('analyst','admin')", name="ck_user_role"),)

    def get_id(self):
        return str(self.user_id)

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)  # scrypt (werkzeug 3 default)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    @property
    def is_admin(self) -> bool:
        return self.role == ROLE_ADMIN


class FlowRecord(db.Model):
    __tablename__ = "flow_record"
    flow_id = db.Column(db.Integer, primary_key=True)
    src_ip = db.Column(db.String(45), index=True)
    dst_ip = db.Column(db.String(45), index=True)
    src_port = db.Column(db.Integer)
    dst_port = db.Column(db.Integer)
    protocol = db.Column(db.Integer)
    features = db.Column(db.JSON, nullable=False, default=dict)
    captured_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    source = db.Column(db.String(16), nullable=False, default="csv")

    __table_args__ = (db.CheckConstraint("source IN ('live','csv','replay')", name="ck_flow_source"),)


class ModelVersion(db.Model):
    __tablename__ = "model_version"
    model_id = db.Column(db.Integer, primary_key=True)
    model_type = db.Column(db.String(32), nullable=False)
    file_path = db.Column(db.String(512), nullable=False, unique=True)
    params = db.Column(db.JSON, nullable=False, default=dict)
    metrics = db.Column(db.JSON, nullable=False, default=dict)
    is_active = db.Column(db.Boolean, nullable=False, default=False)
    trained_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    __table_args__ = (db.CheckConstraint("model_type IN ('isolation_forest','random_forest')",
                                         name="ck_model_type"),)

    @property
    def run_id(self) -> str:
        return (self.params or {}).get("run_id", "")

    @property
    def data_source(self) -> str:
        return (self.params or {}).get("data_source", "unknown")


class Detection(db.Model):
    __tablename__ = "detection"
    detection_id = db.Column(db.Integer, primary_key=True)
    flow_id = db.Column(db.Integer, db.ForeignKey("flow_record.flow_id"), nullable=False, index=True)
    if_model_id = db.Column(db.Integer, db.ForeignKey("model_version.model_id"))
    rf_model_id = db.Column(db.Integer, db.ForeignKey("model_version.model_id"))
    anomaly_score = db.Column(db.Float, nullable=False)
    predicted_class = db.Column(db.String(64), nullable=False, index=True)
    confidence = db.Column(db.Float)
    severity = db.Column(db.String(16))
    status = db.Column(db.String(16), nullable=False, default="new", index=True)
    detected_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)

    flow = db.relationship("FlowRecord", lazy="joined")
    if_model = db.relationship("ModelVersion", foreign_keys=[if_model_id])
    rf_model = db.relationship("ModelVersion", foreign_keys=[rf_model_id])
    actions = db.relationship("ResponseAction", back_populates="detection", lazy="select",
                              order_by="ResponseAction.action_id")

    __table_args__ = (db.CheckConstraint(
        "status IN ('new','acknowledged','resolved','false_positive')", name="ck_detection_status"),)


class ResponsePolicy(db.Model):
    __tablename__ = "response_policy"
    policy_id = db.Column(db.Integer, primary_key=True)
    threat_class = db.Column(db.String(64), nullable=False, index=True)
    severity = db.Column(db.String(16), nullable=False)
    min_confidence = db.Column(db.Float, nullable=False, default=0.0)
    action_type = db.Column(db.String(32), nullable=False)
    duration_min = db.Column(db.Integer, nullable=False, default=0)
    enabled = db.Column(db.Boolean, nullable=False, default=True)


class ResponseAction(db.Model):
    __tablename__ = "response_action"
    action_id = db.Column(db.Integer, primary_key=True)
    detection_id = db.Column(db.Integer, db.ForeignKey("detection.detection_id"), index=True)
    policy_id = db.Column(db.Integer, db.ForeignKey("response_policy.policy_id"))
    approved_by = db.Column(db.Integer, db.ForeignKey("user.user_id"), nullable=True)
    action_type = db.Column(db.String(32), nullable=False)
    target_ip = db.Column(db.String(45), nullable=False, index=True)
    mode = db.Column(db.String(16), nullable=False)
    status = db.Column(db.String(16), nullable=False, index=True)
    executed_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    expires_at = db.Column(db.DateTime, index=True)
    outcome = db.Column(db.Text)

    detection = db.relationship("Detection", back_populates="actions")
    policy = db.relationship("ResponsePolicy")
    approver = db.relationship("User")

    __table_args__ = (
        db.CheckConstraint("mode IN ('enforce','simulate')", name="ck_action_mode"),
        db.CheckConstraint("status IN ('active','expired','released','failed','simulated')",
                           name="ck_action_status"),
    )


class Whitelist(db.Model):
    __tablename__ = "whitelist"
    entry_id = db.Column(db.Integer, primary_key=True)
    ip_or_cidr = db.Column(db.String(64), nullable=False, unique=True)
    reason = db.Column(db.String(255))
    added_at = db.Column(db.DateTime, nullable=False, default=utcnow)


class AuditLog(db.Model):
    __tablename__ = "audit_log"
    log_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.user_id"), nullable=True, index=True)
    event = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)

    user = db.relationship("User")
