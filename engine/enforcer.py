"""Layer 4 - Enforcement adapter (iptables / conntrack).

Two modes behind one interface:

* ``simulate`` - builds the exact argument lists and logs them; nothing is executed.
  This is the only mode that can work on PythonAnywhere (no root, no netfilter).
* ``enforce``  - executes them with ``subprocess.run(argv, shell=False)`` on a Linux
  lab VM where the process has CAP_NET_ADMIN.

Safety rules applied to *every* call, in both modes:
    * the target must be a single, valid, routable IP address (no CIDR, no hostnames,
      no loopback / unspecified / multicast / reserved / link-local addresses);
    * the whitelist is checked before any rule is built;
    * commands are argument lists - never a shell string, never shell=True.

All rules live in a dedicated chain (``HYBRID_IDS``) jumped to from INPUT and FORWARD
and are tagged with an iptables comment, so they can be listed and removed precisely.
"""
from __future__ import annotations

import logging
import re
import shlex
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from engine.policy import (
    ACTION_ALERT, ACTION_QUARANTINE, ACTION_RATE_LIMIT, ACTION_TEMP_BLOCK, ACTION_TERMINATE,
    match_whitelist, parse_ip,
)

log = logging.getLogger("hybrid_ids.enforcer")

MODE_SIMULATE = "simulate"
MODE_ENFORCE = "enforce"
STATUS_ACTIVE = "active"
STATUS_SIMULATED = "simulated"
STATUS_FAILED = "failed"


class EnforcementRefused(ValueError):
    """The request was rejected before any command was built (bad IP, whitelisted...)."""


@dataclass
class EnforcementResult:
    success: bool
    status: str                     # active | simulated | failed
    mode: str
    commands: list[list[str]] = field(default_factory=list)
    output: str = ""

    @property
    def command_text(self) -> str:
        return "\n".join(" ".join(shlex.quote(a) for a in c) for c in self.commands)


def validate_target_ip(value: str) -> str:
    """Return the canonical address or raise EnforcementRefused."""
    try:
        addr = parse_ip(value)
    except ValueError as exc:
        raise EnforcementRefused(f"invalid IP address {value!r}") from exc
    for flag, why in (("is_unspecified", "unspecified"), ("is_loopback", "loopback"),
                      ("is_multicast", "multicast"), ("is_reserved", "reserved"),
                      ("is_link_local", "link-local")):
        if getattr(addr, flag):
            raise EnforcementRefused(f"refusing to act on {why} address {addr}")
    return str(addr)


class BaseEnforcer(ABC):
    def __init__(self, mode: str = MODE_SIMULATE, whitelist_provider=lambda: ()):
        if mode not in (MODE_SIMULATE, MODE_ENFORCE):
            raise ValueError(f"unknown enforcer mode {mode!r}")
        self.mode = mode
        self.whitelist_provider = whitelist_provider

    def check_target(self, ip: str) -> str:
        ip = validate_target_ip(ip)
        entry = match_whitelist(ip, self.whitelist_provider())
        if entry:
            raise EnforcementRefused(f"{ip} is whitelisted ({entry}) - not blocking")
        return ip

    @abstractmethod
    def apply(self, action_type: str, target_ip: str, tag: str) -> EnforcementResult: ...

    @abstractmethod
    def release(self, action_type: str, target_ip: str, tag: str) -> EnforcementResult: ...


class IptablesEnforcer(BaseEnforcer):
    def __init__(self, mode: str = MODE_SIMULATE, whitelist_provider=lambda: (),
                 chain: str = "HYBRID_IDS", rate: str = "20/second", burst: int = 40,
                 runner=subprocess.run, timeout: int = 10):
        super().__init__(mode, whitelist_provider)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,27}", chain):
            raise ValueError("invalid chain name")
        if not re.fullmatch(r"\d+/(second|minute|hour)", rate):
            raise ValueError("invalid rate specification")
        self.chain, self.rate, self.burst = chain, rate, int(burst)
        self.runner, self.timeout = runner, timeout
        self._chain_ready = False

    # ------------------------------------------------------------- command builders
    @staticmethod
    def _tag(tag: str) -> str:
        clean = re.sub(r"[^A-Za-z0-9:_-]", "", str(tag))[:48]
        return f"hybrid-ids:{clean}" if clean else "hybrid-ids"

    @staticmethod
    def _tables(ip: str) -> str:
        return "ip6tables" if ":" in ip else "iptables"

    def _rule_specs(self, action_type: str, ip: str, tag: str) -> list[list[str]]:
        comment = ["-m", "comment", "--comment", self._tag(tag)]
        if action_type == ACTION_TEMP_BLOCK:
            return [["-s", ip, *comment, "-j", "DROP"]]
        if action_type == ACTION_RATE_LIMIT:
            name = ("hids" + re.sub(r"[^A-Za-z0-9]", "", str(tag)))[:15]
            return [["-s", ip, "-m", "hashlimit", "--hashlimit-above", self.rate,
                     "--hashlimit-burst", str(self.burst), "--hashlimit-mode", "srcip",
                     "--hashlimit-name", name, *comment, "-j", "DROP"]]
        if action_type == ACTION_QUARANTINE:
            return [["-s", ip, *comment, "-j", "DROP"], ["-d", ip, *comment, "-j", "DROP"]]
        return []

    def build_apply_commands(self, action_type: str, ip: str, tag: str) -> list[list[str]]:
        tables = self._tables(ip)
        cmds = [[tables, "-w", "-I", self.chain, *spec] for spec in self._rule_specs(action_type, ip, tag)]
        if action_type == ACTION_TERMINATE:
            family = ["-f", "ipv6"] if ":" in ip else []
            cmds += [["conntrack", *family, "-D", "-s", ip], ["conntrack", *family, "-D", "-d", ip]]
        return cmds

    def build_release_commands(self, action_type: str, ip: str, tag: str) -> list[list[str]]:
        tables = self._tables(ip)
        return [[tables, "-w", "-D", self.chain, *spec] for spec in self._rule_specs(action_type, ip, tag)]

    def chain_setup_commands(self, v6: bool = False) -> list[list[str]]:
        t = "ip6tables" if v6 else "iptables"
        return [[t, "-w", "-N", self.chain],
                [t, "-w", "-I", "INPUT", "-j", self.chain],
                [t, "-w", "-I", "FORWARD", "-j", self.chain]]

    # --------------------------------------------------------------------- execution
    @staticmethod
    def available() -> bool:
        return shutil.which("iptables") is not None

    def _run(self, cmds: list[list[str]]) -> tuple[bool, str]:
        outputs = []
        for cmd in cmds:
            exe = shutil.which(cmd[0])
            if exe is None:
                return False, f"{cmd[0]} not found - enforce mode needs a Linux host with netfilter tools"
            try:
                proc = self.runner([exe, *cmd[1:]], shell=False, capture_output=True, text=True,
                                   timeout=self.timeout, check=False)
            except (OSError, subprocess.SubprocessError) as exc:
                return False, f"{cmd[0]} failed: {exc}"
            text = (proc.stdout or "") + (proc.stderr or "")
            outputs.append(text.strip())
            # conntrack exits 1 when no entries matched - that is not an error for us.
            if proc.returncode != 0 and not (cmd[0] == "conntrack" and "0 flow entries" in text):
                return False, f"{' '.join(cmd)} -> exit {proc.returncode}: {text.strip()}"
        return True, "\n".join(o for o in outputs if o)

    def _ensure_chain(self, ip: str):
        if self._chain_ready or self.mode != MODE_ENFORCE:
            return
        t = self._tables(ip)
        exe = shutil.which(t)
        if exe is None:
            return
        check = self.runner([exe, "-w", "-L", self.chain, "-n"], shell=False, capture_output=True,
                            text=True, timeout=self.timeout, check=False)
        if check.returncode != 0:
            self._run(self.chain_setup_commands(v6=(t == "ip6tables")))
        self._chain_ready = True

    def apply(self, action_type: str, target_ip: str, tag: str) -> EnforcementResult:
        if action_type == ACTION_ALERT:
            return EnforcementResult(True, STATUS_SIMULATED if self.mode == MODE_SIMULATE else STATUS_ACTIVE,
                                     self.mode, [], "alert only - no network change")
        try:
            ip = self.check_target(target_ip)
        except EnforcementRefused as exc:
            log.warning("enforcement refused: %s", exc)
            return EnforcementResult(False, STATUS_FAILED, self.mode, [], f"REFUSED: {exc}")
        cmds = self.build_apply_commands(action_type, ip, tag)
        if not cmds:
            return EnforcementResult(False, STATUS_FAILED, self.mode, [], f"unknown action {action_type!r}")
        result = EnforcementResult(True, STATUS_SIMULATED, self.mode, cmds)
        if self.mode == MODE_SIMULATE:
            result.output = "SIMULATED - would run:\n" + result.command_text
            log.info("[simulate] %s", result.command_text.replace("\n", " ; "))
            return result
        self._ensure_chain(ip)
        ok, out = self._run(cmds)
        result.success, result.output = ok, out or "ok"
        result.status = STATUS_ACTIVE if ok else STATUS_FAILED
        log.info("[enforce] %s -> %s", result.command_text.replace("\n", " ; "), result.status)
        return result

    def release(self, action_type: str, target_ip: str, tag: str) -> EnforcementResult:
        try:
            ip = validate_target_ip(target_ip)  # no whitelist check: releasing is always allowed
        except EnforcementRefused as exc:
            return EnforcementResult(False, STATUS_FAILED, self.mode, [], f"REFUSED: {exc}")
        cmds = self.build_release_commands(action_type, ip, tag)
        result = EnforcementResult(True, STATUS_SIMULATED, self.mode, cmds)
        if not cmds:
            result.output = "nothing to undo"
            return result
        if self.mode == MODE_SIMULATE:
            result.output = "SIMULATED - would run:\n" + result.command_text
            log.info("[simulate] release %s", result.command_text.replace("\n", " ; "))
            return result
        ok, out = self._run(cmds)
        result.success, result.output = ok, out or "ok"
        result.status = STATUS_ACTIVE if ok else STATUS_FAILED
        return result
