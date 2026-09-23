"""SOC alert notifier.

Every alert is always written to the application log (and, by the service layer, to the
audit_log table and the dashboard). E-mail via SMTP is optional: configure SMTP_HOST etc.
Failures to e-mail never break detection - they are reported back to the caller.

Note: PythonAnywhere free accounts can only reach an allow-list of external hosts, so
SMTP may be unavailable there; the dashboard remains the primary alert channel.
"""
from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage

log = logging.getLogger("hybrid_ids.notifier")


@dataclass
class SmtpSettings:
    host: str = ""
    port: int = 587
    user: str = ""
    password: str = ""
    to: str = ""
    sender: str = "hybrid-ids@localhost"

    @property
    def enabled(self) -> bool:
        return bool(self.host and self.to)


class Notifier:
    def __init__(self, smtp: SmtpSettings | None = None, sender=smtplib.SMTP):
        self.smtp = smtp or SmtpSettings()
        self._smtp_cls = sender

    def send(self, subject: str, body: str, urgent: bool = False) -> tuple[list[str], str | None]:
        """Returns (channels delivered to, error message or None)."""
        prefix = "[URGENT] " if urgent else ""
        (log.critical if urgent else log.warning)("SOC ALERT %s%s | %s", prefix, subject,
                                                  body.replace("\n", " | "))
        channels, error = ["log"], None
        if self.smtp.enabled:
            try:
                msg = EmailMessage()
                msg["Subject"] = f"{prefix}[Hybrid IDS] {subject}"
                msg["From"] = self.smtp.user or self.smtp.sender
                msg["To"] = self.smtp.to
                if urgent:
                    msg["X-Priority"] = "1"
                msg.set_content(body)
                with self._smtp_cls(self.smtp.host, self.smtp.port, timeout=10) as s:
                    s.starttls()
                    if self.smtp.user:
                        s.login(self.smtp.user, self.smtp.password)
                    s.send_message(msg)
                channels.append("email")
            except Exception as exc:  # noqa: BLE001 - notification must never break detection
                error = f"e-mail failed: {exc}"
                log.error(error)
        return channels, error
