"""Layer 4 - Policy engine.

Maps (threat class, severity, confidence) to response actions:

1. Rule selection - among the *enabled* rules for the threat class whose
   ``min_confidence <= p``, the highest-confidence tier wins and every rule in that tier
   is executed (e.g. DoS/DDoS at p = 0.95 -> tier 0.90 -> temporary block 60 min + alert;
   at p = 0.75 -> tier 0.70 -> rate-limit 30 min + alert).
   No matching tier -> alert only.
2. Safety invariant - "Unknown anomaly" never triggers an automatic block, whatever
   the policy table says.
3. Target - blocking actions hit the remote / attacking endpoint; quarantine hits the
   internal host (see ``resolve_target``).
4. Whitelist - a blocking action against a whitelisted address is not executed (the
   enforcer re-checks as a second line of defence).
5. Escalation - each earlier action of the same kind (e.g. a previous temporary block)
   against the same address within 24 h doubles the duration, capped at 24 h.

Framework-independent: the Flask service layer feeds it rules and whitelist entries from
the database.
"""
from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field

from engine.schema import (
    CAT_BOT, CAT_BRUTE, CAT_DOS, CAT_OTHER, CAT_PHISHING, CAT_PORTSCAN, CAT_WEB,
    CATEGORY_SEVERITY, UNKNOWN_ANOMALY,
)

ACTION_ALERT = "alert"
ACTION_RATE_LIMIT = "rate_limit"
ACTION_TEMP_BLOCK = "temp_block"
ACTION_TERMINATE = "terminate_session"
ACTION_QUARANTINE = "quarantine_host"
ACTION_TYPES = [ACTION_ALERT, ACTION_RATE_LIMIT, ACTION_TEMP_BLOCK, ACTION_TERMINATE, ACTION_QUARANTINE]
BLOCKING_ACTIONS = {ACTION_RATE_LIMIT, ACTION_TEMP_BLOCK, ACTION_TERMINATE, ACTION_QUARANTINE}
TIMED_ACTIONS = {ACTION_RATE_LIMIT, ACTION_TEMP_BLOCK, ACTION_QUARANTINE}
ACTION_LABELS = {
    ACTION_ALERT: "Alert",
    ACTION_RATE_LIMIT: "Rate-limit",
    ACTION_TEMP_BLOCK: "Temporary block",
    ACTION_TERMINATE: "Terminate session",
    ACTION_QUARANTINE: "Quarantine host",
}
SEVERITIES = ["Low", "Medium", "Medium-High", "High", "Critical"]

MAX_DURATION_MIN = 24 * 60


@dataclass(frozen=True)
class PolicyRule:
    threat_class: str
    severity: str
    min_confidence: float
    action_type: str
    duration_min: int = 0
    enabled: bool = True
    policy_id: int | None = None


# The default policy table from the design (one row per action; rows sharing a class and
# min_confidence form one tier). duration 0 = no expiry (alerts, session kills, and
# quarantine, which an analyst releases manually).
DEFAULT_POLICY_TABLE = [
    PolicyRule(CAT_DOS, "High", 0.90, ACTION_TEMP_BLOCK, 60),
    PolicyRule(CAT_DOS, "High", 0.90, ACTION_ALERT, 0),
    PolicyRule(CAT_DOS, "High", 0.70, ACTION_RATE_LIMIT, 30),
    PolicyRule(CAT_DOS, "High", 0.70, ACTION_ALERT, 0),
    PolicyRule(CAT_BRUTE, "High", 0.85, ACTION_TERMINATE, 0),
    PolicyRule(CAT_BRUTE, "High", 0.85, ACTION_TEMP_BLOCK, 60),
    PolicyRule(CAT_BRUTE, "High", 0.85, ACTION_ALERT, 0),
    PolicyRule(CAT_WEB, "High", 0.85, ACTION_TERMINATE, 0),
    PolicyRule(CAT_WEB, "High", 0.85, ACTION_TEMP_BLOCK, 30),
    PolicyRule(CAT_WEB, "High", 0.85, ACTION_ALERT, 0),
    PolicyRule(CAT_BOT, "Critical", 0.85, ACTION_QUARANTINE, 0),
    PolicyRule(CAT_BOT, "Critical", 0.85, ACTION_TEMP_BLOCK, 60),  # block the controller
    PolicyRule(CAT_BOT, "Critical", 0.85, ACTION_ALERT, 0),        # urgent (Critical)
    PolicyRule(CAT_PORTSCAN, "Medium", 0.80, ACTION_RATE_LIMIT, 15),
    PolicyRule(CAT_PORTSCAN, "Medium", 0.80, ACTION_ALERT, 0),
    PolicyRule(CAT_OTHER, "Medium-High", 0.0, ACTION_ALERT, 0),
    PolicyRule(UNKNOWN_ANOMALY, "Medium-High", 0.0, ACTION_ALERT, 0),
    PolicyRule(CAT_PHISHING, "Medium", 0.0, ACTION_ALERT, 0),
]


@dataclass
class PlannedAction:
    action_type: str
    target_ip: str
    duration_min: int
    severity: str
    policy_id: int | None
    urgent: bool = False
    reason: str = ""
    skipped_reason: str | None = None  # set when the action must NOT be executed

    @property
    def is_blocking(self) -> bool:
        return self.action_type in BLOCKING_ACTIONS


@dataclass
class PolicyDecision:
    threat_class: str
    confidence: float
    severity: str
    tier: float | None
    actions: list[PlannedAction] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ------------------------------------------------------------------------ helpers
def parse_ip(value: str):
    """Parse a single IP address (no CIDR, no hostnames). Raises ValueError."""
    if value is None:
        raise ValueError("no IP address")
    text = str(value).strip()
    if not text or "/" in text:
        raise ValueError(f"not a single IP address: {value!r}")
    return ipaddress.ip_address(text)


def normalise_whitelist_entry(value: str) -> str:
    """Validate an IP or CIDR for the whitelist and return its canonical form."""
    text = str(value).strip()
    if not text:
        raise ValueError("empty whitelist entry")
    net = ipaddress.ip_network(text, strict=False)
    return str(net.network_address) if net.num_addresses == 1 else str(net)


def match_whitelist(ip: str, entries) -> str | None:
    """Return the whitelist entry covering ``ip``, or None. Unparseable entries are ignored."""
    try:
        addr = parse_ip(ip)
    except ValueError:
        return None
    for entry in entries or ():
        try:
            if addr in ipaddress.ip_network(str(entry).strip(), strict=False):
                return str(entry).strip()
        except ValueError:
            continue
    return None


def _is_internal(ip: str) -> bool:
    try:
        return parse_ip(ip).is_private
    except ValueError:
        return False


def resolve_target(action_type: str, src_ip: str, dst_ip: str) -> str:
    """Which endpoint an action applies to.

    * quarantine_host -> the internal (private) host, normally the source
      (a bot beaconing out to its controller).
    * everything else -> the remote party: the destination when an internal host talks
      to a public address (bot -> C2, infiltration call-back), otherwise the source
      (external or NAT'd attacker -> victim server).
    """
    src_internal, dst_internal = _is_internal(src_ip), _is_internal(dst_ip)
    if action_type == ACTION_QUARANTINE:
        if src_internal or not dst_internal:
            return src_ip
        return dst_ip
    if src_internal and dst_ip and not dst_internal:
        return dst_ip
    return src_ip


def escalated_duration(base_min: int, prior_offences: int, cap_min: int = MAX_DURATION_MIN) -> int:
    """Double the duration for each earlier offence within the window, capped."""
    if base_min <= 0:
        return base_min
    return int(min(cap_min, base_min * (2 ** max(0, prior_offences))))


def select_rules(rules, threat_class: str, confidence: float) -> tuple[float | None, list[PolicyRule]]:
    """Highest enabled tier with min_confidence <= confidence for this class."""
    eligible = [r for r in rules if r.enabled and r.threat_class == threat_class
                and r.min_confidence <= confidence + 1e-12]
    if not eligible:
        return None, []
    tier = max(r.min_confidence for r in eligible)
    return tier, [r for r in eligible if abs(r.min_confidence - tier) < 1e-9]


def plan_response(rules, threat_class: str, confidence: float, src_ip: str, dst_ip: str,
                  whitelist=(), prior_offences=lambda ip, action_type: 0, severity: str | None = None,
                  cap_min: int = MAX_DURATION_MIN) -> PolicyDecision:
    """Decide which actions to take for one detection.

    ``prior_offences(ip, action_type)`` returns how many earlier actions of that type
    targeted ``ip`` within the repeat-offence window.
    """
    severity = severity or CATEGORY_SEVERITY.get(threat_class, "Medium")
    tier, chosen = select_rules(rules, threat_class, confidence)
    decision = PolicyDecision(threat_class, confidence, severity, tier)
    if not chosen:
        decision.notes.append("no policy tier matched - alert only")
        chosen = [PolicyRule(threat_class, severity, 0.0, ACTION_ALERT, 0)]

    if threat_class == UNKNOWN_ANOMALY and any(r.action_type in BLOCKING_ACTIONS for r in chosen):
        decision.notes.append("blocking rules ignored: Unknown anomalies are never auto-blocked")
        chosen = [r for r in chosen if r.action_type not in BLOCKING_ACTIONS] or \
            [PolicyRule(threat_class, severity, 0.0, ACTION_ALERT, 0)]

    for rule in chosen:
        sev = rule.severity or severity
        target = resolve_target(rule.action_type, src_ip, dst_ip)
        duration = rule.duration_min
        reason = f"{threat_class} p={confidence:.2f} >= tier {rule.min_confidence:.2f}"
        skipped = None
        if rule.action_type in BLOCKING_ACTIONS:
            wl = match_whitelist(target, whitelist)
            if wl:
                skipped = f"target {target} is whitelisted ({wl})"
            else:
                try:
                    parse_ip(target)
                except ValueError:
                    skipped = f"invalid target address {target!r}"
            if skipped is None and rule.action_type in TIMED_ACTIONS and duration > 0:
                prior = prior_offences(target, rule.action_type)
                new = escalated_duration(duration, prior, cap_min)
                if new != duration:
                    reason += f"; repeat offence x{prior} -> {duration} min doubled to {new} min"
                duration = new
        decision.actions.append(PlannedAction(
            action_type=rule.action_type, target_ip=target, duration_min=duration, severity=sev,
            policy_id=rule.policy_id, urgent=(sev == "Critical"), reason=reason,
            skipped_reason=skipped))
    return decision
