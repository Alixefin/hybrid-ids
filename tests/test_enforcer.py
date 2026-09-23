"""Enforcer: simulate mode, whitelist and invalid-IP handling, command construction."""
import subprocess

import pytest

from engine import enforcer as enf_mod
from engine.enforcer import (
    MODE_ENFORCE, MODE_SIMULATE, STATUS_ACTIVE, STATUS_FAILED, STATUS_SIMULATED,
    EnforcementRefused, IptablesEnforcer, validate_target_ip,
)
from engine.policy import (
    ACTION_ALERT, ACTION_QUARANTINE, ACTION_RATE_LIMIT, ACTION_TEMP_BLOCK, ACTION_TERMINATE,
)


class Recorder:
    """Stands in for subprocess.run and records every call."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.calls, self.rc, self.out, self.err = [], returncode, stdout, stderr

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, self.rc, self.out, self.err)


@pytest.fixture
def sim():
    runner = Recorder()
    return IptablesEnforcer(MODE_SIMULATE, whitelist_provider=lambda: ["10.0.0.0/8", "192.168.10.3"],
                            runner=runner), runner


@pytest.mark.parametrize("action", [ACTION_TEMP_BLOCK, ACTION_RATE_LIMIT, ACTION_TERMINATE, ACTION_QUARANTINE])
def test_simulate_mode_builds_commands_but_never_executes(sim, action):
    e, runner = sim
    r = e.apply(action, "172.16.0.1", "abc123")
    assert r.success and r.status == STATUS_SIMULATED and r.mode == MODE_SIMULATE
    assert r.commands and all(isinstance(c, list) and all(isinstance(a, str) for a in c) for c in r.commands)
    assert "SIMULATED" in r.output
    assert runner.calls == []


def test_simulate_release_never_executes(sim):
    e, runner = sim
    r = e.release(ACTION_TEMP_BLOCK, "172.16.0.1", "abc123")
    assert r.status == STATUS_SIMULATED and r.commands[0][2] == "-D" and runner.calls == []


@pytest.mark.parametrize("ip", ["10.1.2.3", "10.255.255.254", "192.168.10.3"])
def test_whitelisted_targets_are_refused(sim, ip):
    e, runner = sim
    for action in (ACTION_TEMP_BLOCK, ACTION_RATE_LIMIT, ACTION_TERMINATE, ACTION_QUARANTINE):
        r = e.apply(action, ip, "t")
        assert not r.success and r.status == STATUS_FAILED
        assert "whitelisted" in r.output and r.commands == []
    assert runner.calls == []


@pytest.mark.parametrize("ip", [
    "", None, "999.1.1.1", "1.2.3", "10.0.0.0/24", "example.com", "1.2.3.4; rm -rf /",
    "1.2.3.4 -j ACCEPT", "$(reboot)", "127.0.0.1", "0.0.0.0", "::1", "224.0.0.1",
    "255.255.255.255", "169.254.1.1", "fe80::1",
])
def test_invalid_or_dangerous_ips_are_refused(sim, ip):
    e, runner = sim
    r = e.apply(ACTION_TEMP_BLOCK, ip, "t")
    assert not r.success and r.status == STATUS_FAILED and r.output.startswith("REFUSED")
    assert r.commands == [] and runner.calls == []


def test_validate_target_ip_normalises():
    assert validate_target_ip(" 172.16.0.1 ") == "172.16.0.1"
    assert validate_target_ip("2001:DB8::1") == "2001:db8::1"
    with pytest.raises(EnforcementRefused):
        validate_target_ip("127.0.0.1")


def test_alert_needs_no_command(sim):
    e, runner = sim
    r = e.apply(ACTION_ALERT, "10.1.2.3", "t")  # alerts are allowed even for whitelisted hosts
    assert r.success and r.commands == [] and runner.calls == []


def test_command_shapes():
    e = IptablesEnforcer(MODE_SIMULATE)
    block = e.build_apply_commands(ACTION_TEMP_BLOCK, "172.16.0.1", "a1")
    assert block == [["iptables", "-w", "-I", "HYBRID_IDS", "-s", "172.16.0.1", "-m", "comment",
                      "--comment", "hybrid-ids:a1", "-j", "DROP"]]
    rl = e.build_apply_commands(ACTION_RATE_LIMIT, "172.16.0.1", "a1")[0]
    assert "hashlimit" in rl and rl[rl.index("--hashlimit-above") + 1] == "20/second"
    kill = e.build_apply_commands(ACTION_TERMINATE, "172.16.0.1", "a1")
    assert kill == [["conntrack", "-D", "-s", "172.16.0.1"], ["conntrack", "-D", "-d", "172.16.0.1"]]
    q = e.build_apply_commands(ACTION_QUARANTINE, "192.168.10.5", "a1")
    assert [c[4] for c in q] == ["-s", "-d"]
    v6 = e.build_apply_commands(ACTION_TEMP_BLOCK, "2001:db8::1", "a1")
    assert v6[0][0] == "ip6tables"
    assert e.build_apply_commands(ACTION_TERMINATE, "2001:db8::1", "a1")[0][:3] == ["conntrack", "-f", "ipv6"]


def test_tag_is_sanitised():
    e = IptablesEnforcer(MODE_SIMULATE)
    cmd = e.build_apply_commands(ACTION_TEMP_BLOCK, "172.16.0.1", 'x"; rm -rf / #')[0]
    assert cmd[cmd.index("--comment") + 1] == "hybrid-ids:xrm-rf"


def test_constructor_validates_chain_and_rate():
    with pytest.raises(ValueError):
        IptablesEnforcer(chain="INPUT; reboot")
    with pytest.raises(ValueError):
        IptablesEnforcer(rate="lots")
    with pytest.raises(ValueError):
        IptablesEnforcer(mode="yolo")


def test_enforce_mode_runs_argument_lists_without_shell(monkeypatch):
    monkeypatch.setattr(enf_mod.shutil, "which", lambda name: f"/usr/sbin/{name}")
    runner = Recorder()
    e = IptablesEnforcer(MODE_ENFORCE, runner=runner)
    e._chain_ready = True
    r = e.apply(ACTION_TERMINATE, "172.16.0.1", "t1")
    assert r.success and r.status == STATUS_ACTIVE
    assert len(runner.calls) == 2
    for args, kwargs in runner.calls:
        assert isinstance(args, list) and kwargs["shell"] is False and kwargs["timeout"] > 0
        assert args[0] == "/usr/sbin/conntrack"


def test_enforce_mode_failure_is_reported(monkeypatch):
    monkeypatch.setattr(enf_mod.shutil, "which", lambda name: f"/usr/sbin/{name}")
    e = IptablesEnforcer(MODE_ENFORCE, runner=Recorder(returncode=4, stderr="Permission denied (you must be root)"))
    e._chain_ready = True
    r = e.apply(ACTION_TEMP_BLOCK, "172.16.0.1", "t1")
    assert not r.success and r.status == STATUS_FAILED and "Permission denied" in r.output


def test_enforce_mode_conntrack_no_match_is_not_an_error(monkeypatch):
    monkeypatch.setattr(enf_mod.shutil, "which", lambda name: f"/usr/sbin/{name}")
    runner = Recorder(returncode=1, stderr="conntrack v1.4.6 (conntrack-tools): 0 flow entries have been deleted.")
    e = IptablesEnforcer(MODE_ENFORCE, runner=runner)
    assert e.apply(ACTION_TERMINATE, "172.16.0.1", "t").success


def test_enforce_mode_without_iptables_fails_cleanly(monkeypatch):
    monkeypatch.setattr(enf_mod.shutil, "which", lambda name: None)
    runner = Recorder()
    e = IptablesEnforcer(MODE_ENFORCE, runner=runner)
    r = e.apply(ACTION_TEMP_BLOCK, "172.16.0.1", "t")
    assert not r.success and "not found" in r.output and runner.calls == []


def test_enforce_mode_still_refuses_whitelisted(monkeypatch):
    monkeypatch.setattr(enf_mod.shutil, "which", lambda name: f"/usr/sbin/{name}")
    runner = Recorder()
    e = IptablesEnforcer(MODE_ENFORCE, whitelist_provider=lambda: ["172.16.0.0/12"], runner=runner)
    r = e.apply(ACTION_TEMP_BLOCK, "172.16.0.1", "t")
    assert not r.success and "whitelisted" in r.output and runner.calls == []
