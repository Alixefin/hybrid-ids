"""Policy engine rule selection, targets, whitelist and escalation."""
import pytest

from engine.policy import (
    ACTION_ALERT, ACTION_QUARANTINE, ACTION_RATE_LIMIT, ACTION_TEMP_BLOCK, ACTION_TERMINATE,
    DEFAULT_POLICY_TABLE, PolicyRule, escalated_duration, match_whitelist,
    normalise_whitelist_entry, plan_response, resolve_target, select_rules,
)
from engine.schema import (
    CAT_BOT, CAT_BRUTE, CAT_DOS, CAT_OTHER, CAT_PORTSCAN, CAT_WEB, UNKNOWN_ANOMALY,
)

RULES = DEFAULT_POLICY_TABLE
ATTACKER, VICTIM = "172.16.0.1", "192.168.10.50"


def kinds(decision):
    return sorted((a.action_type, a.duration_min) for a in decision.actions)


@pytest.mark.parametrize("cls,p,expected", [
    (CAT_DOS, 0.95, [(ACTION_ALERT, 0), (ACTION_TEMP_BLOCK, 60)]),
    (CAT_DOS, 0.90, [(ACTION_ALERT, 0), (ACTION_TEMP_BLOCK, 60)]),      # boundary: >= 0.90
    (CAT_DOS, 0.89, [(ACTION_ALERT, 0), (ACTION_RATE_LIMIT, 30)]),
    (CAT_DOS, 0.70, [(ACTION_ALERT, 0), (ACTION_RATE_LIMIT, 30)]),
    (CAT_BRUTE, 0.85, [(ACTION_ALERT, 0), (ACTION_TEMP_BLOCK, 60), (ACTION_TERMINATE, 0)]),
    (CAT_WEB, 0.99, [(ACTION_ALERT, 0), (ACTION_TEMP_BLOCK, 30), (ACTION_TERMINATE, 0)]),
    (CAT_PORTSCAN, 0.80, [(ACTION_ALERT, 0), (ACTION_RATE_LIMIT, 15)]),
    (CAT_OTHER, 0.99, [(ACTION_ALERT, 0)]),
    (UNKNOWN_ANOMALY, 0.40, [(ACTION_ALERT, 0)]),
])
def test_default_table_selection(cls, p, expected):
    assert kinds(plan_response(RULES, cls, p, ATTACKER, VICTIM)) == expected


@pytest.mark.parametrize("cls,p", [(CAT_BRUTE, 0.84), (CAT_WEB, 0.70), (CAT_PORTSCAN, 0.79), (CAT_BOT, 0.80)])
def test_below_every_tier_falls_back_to_alert_only(cls, p):
    d = plan_response(RULES, cls, p, ATTACKER, VICTIM)
    assert kinds(d) == [(ACTION_ALERT, 0)]
    assert d.tier is None and "alert only" in d.notes[0]


def test_select_rules_picks_highest_tier_only():
    tier, rules = select_rules(RULES, CAT_DOS, 0.97)
    assert tier == 0.90 and {r.action_type for r in rules} == {ACTION_TEMP_BLOCK, ACTION_ALERT}


def test_disabled_rules_are_ignored():
    rules = [r if not (r.threat_class == CAT_DOS and r.min_confidence == 0.90)
             else PolicyRule(r.threat_class, r.severity, r.min_confidence, r.action_type, r.duration_min, False)
             for r in RULES]
    assert kinds(plan_response(rules, CAT_DOS, 0.95, ATTACKER, VICTIM)) == [(ACTION_ALERT, 0), (ACTION_RATE_LIMIT, 30)]


def test_botnet_quarantines_host_and_blocks_controller():
    bot, c2 = "192.168.10.5", "205.174.165.73"
    d = plan_response(RULES, CAT_BOT, 0.9, bot, c2)
    targets = {a.action_type: a.target_ip for a in d.actions}
    assert targets[ACTION_QUARANTINE] == bot
    assert targets[ACTION_TEMP_BLOCK] == c2
    assert all(a.urgent for a in d.actions)          # Critical severity -> urgent alert
    assert d.severity == "Critical"


def test_unknown_anomaly_never_auto_blocks_even_if_misconfigured():
    rules = list(RULES) + [PolicyRule(UNKNOWN_ANOMALY, "High", 0.0, ACTION_TEMP_BLOCK, 60)]
    d = plan_response(rules, UNKNOWN_ANOMALY, 0.5, ATTACKER, VICTIM)
    assert kinds(d) == [(ACTION_ALERT, 0)]
    assert any("never auto-blocked" in n for n in d.notes)


def test_whitelisted_target_is_skipped_but_alert_kept():
    d = plan_response(RULES, CAT_DOS, 0.95, ATTACKER, VICTIM, whitelist=["172.16.0.0/12"])
    block = next(a for a in d.actions if a.action_type == ACTION_TEMP_BLOCK)
    alert = next(a for a in d.actions if a.action_type == ACTION_ALERT)
    assert block.skipped_reason and "whitelisted" in block.skipped_reason
    assert alert.skipped_reason is None


def test_invalid_target_is_skipped():
    d = plan_response(RULES, CAT_DOS, 0.95, "not-an-ip", VICTIM)
    block = next(a for a in d.actions if a.action_type == ACTION_TEMP_BLOCK)
    assert "invalid" in block.skipped_reason


def test_repeat_offences_double_duration_and_cap():
    for prior, expected in [(0, 60), (1, 120), (2, 240), (4, 960), (5, 1440), (9, 1440)]:
        d = plan_response(RULES, CAT_DOS, 0.95, ATTACKER, VICTIM, prior_offences=lambda ip, action, n=prior: n)
        block = next(a for a in d.actions if a.action_type == ACTION_TEMP_BLOCK)
        assert block.duration_min == expected
    assert escalated_duration(0, 3) == 0            # untimed actions never escalate
    assert escalated_duration(15, 1, cap_min=20) == 20


def test_escalation_counts_per_target():
    offences = {(ATTACKER, ACTION_RATE_LIMIT): 2, (ATTACKER, ACTION_TEMP_BLOCK): 5}
    d = plan_response(RULES, CAT_PORTSCAN, 0.9, ATTACKER, VICTIM, prior_offences=lambda ip, action: offences.get((ip, action), 0))
    rl = next(a for a in d.actions if a.action_type == ACTION_RATE_LIMIT)
    assert rl.duration_min == 60 and "doubled" in rl.reason


@pytest.mark.parametrize("action,src,dst,expected", [
    (ACTION_TEMP_BLOCK, "172.16.0.1", "192.168.10.50", "172.16.0.1"),      # NAT'd attacker -> server
    (ACTION_TEMP_BLOCK, "8.8.8.8", "192.168.10.50", "8.8.8.8"),            # external attacker
    (ACTION_TEMP_BLOCK, "192.168.10.8", "205.174.165.73", "205.174.165.73"),  # internal -> C2
    (ACTION_QUARANTINE, "192.168.10.8", "205.174.165.73", "192.168.10.8"),
    (ACTION_QUARANTINE, "205.174.165.73", "192.168.10.8", "192.168.10.8"),
])
def test_resolve_target(action, src, dst, expected):
    assert resolve_target(action, src, dst) == expected


def test_whitelist_helpers():
    assert normalise_whitelist_entry(" 10.1.2.3 ") == "10.1.2.3"
    assert normalise_whitelist_entry("10.1.2.3/8") == "10.0.0.0/8"
    assert normalise_whitelist_entry("2001:db8::/32") == "2001:db8::/32"
    for bad in ["", "10.0.0.300", "example.com", "10.0.0.0/33"]:
        with pytest.raises(ValueError):
            normalise_whitelist_entry(bad)
    assert match_whitelist("10.9.9.9", ["garbage", "10.0.0.0/8"]) == "10.0.0.0/8"
    assert match_whitelist("11.0.0.1", ["10.0.0.0/8"]) is None
    assert match_whitelist("bogus", ["0.0.0.0/0"]) is None
