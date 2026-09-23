"""Generate a synthetic CICIDS2017-schema flow CSV.

Stand-in for the real dataset so the full training / evaluation / dashboard pipeline can
be built and smoke-tested before CICIDS2017 is available.

What it reproduces from the real data
-------------------------------------
* Same 85-column "TrafficLabelling" layout (6 identifier columns + 78 CICFlowMeter
  features + Label), same raw header quirks (leading spaces, the duplicated
  "Fwd Header Length" column), same 15 labels.
* Realistic class imbalance (real CICIDS2017 proportions, with a small floor so the
  rarest classes - Heartbleed, SQL injection, Infiltration - still exist).
* Features are *derived* from per-flow primitives (packet counts, payload sizes,
  duration, flags) the way CICFlowMeter computes them, so the redundant / correlated /
  near-constant columns behave like the real ones, and zero-duration flows produce the
  same NaN / Infinity values in "Flow Bytes/s" and "Flow Packets/s".
* Attack traffic sits inside the published attack time windows of the capture week
  (3-7 July 2017) with the published attacker / victim addresses, so the temporal
  evaluation protocol behaves like it would on the real data.
* Dirt: duplicate rows and a fraction of "hard" attack flows that look benign, plus a
  heavy-tailed benign minority, so results are *not* trivially perfect.

THE NUMBERS PRODUCED FROM THIS DATA ARE NOT REAL EVALUATION RESULTS.

Usage
-----
    python -m training.generate_synthetic_flows                    # 120k rows
    python -m training.generate_synthetic_flows --rows 300000 --seed 7
    python -m training.generate_synthetic_flows --per-day          # one CSV per weekday
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.schema import FEATURE_COLUMNS, RAW_COLUMN_ORDER  # noqa: E402

SYNTHETIC_BANNER = (
    "\n" + "!" * 78 + "\n"
    "!!  SYNTHETIC DATA - NOT CICIDS2017.  Flows are randomly generated to mimic the\n"
    "!!  CICIDS2017 schema. Any metric computed on them is a pipeline smoke test,\n"
    "!!  NOT an evaluation result. Swap in the real dataset before reporting numbers.\n"
    + "!" * 78 + "\n"
)

# Flow counts per label in the full CICIDS2017 (MachineLearningCVE) release.
REAL_LABEL_COUNTS = {
    "BENIGN": 2_273_097,
    "DoS Hulk": 231_073,
    "PortScan": 158_930,
    "DDoS": 128_027,
    "DoS GoldenEye": 10_293,
    "FTP-Patator": 7_938,
    "SSH-Patator": 5_897,
    "DoS slowloris": 5_796,
    "DoS Slowhttptest": 5_499,
    "Bot": 1_966,
    "Web Attack - Brute Force": 1_507,
    "Web Attack - XSS": 652,
    "Infiltration": 36,
    "Web Attack - Sql Injection": 21,
    "Heartbleed": 11,
}

# Raw CSV label spelling. The official files contain a non-ASCII dash in the web-attack
# labels; we write it the way it appears when the files are read as UTF-8 so that the
# normalisation in engine.schema.canonical_label is exercised.
RAW_LABEL_SPELLING = {
    "Web Attack - Brute Force": "Web Attack – Brute Force",
    "Web Attack - XSS": "Web Attack – XSS",
    "Web Attack - Sql Injection": "Web Attack – Sql Injection",
}

# Capture week and published attack windows (local time, 24h).
DAYS = {
    "Monday": datetime(2017, 7, 3),
    "Tuesday": datetime(2017, 7, 4),
    "Wednesday": datetime(2017, 7, 5),
    "Thursday": datetime(2017, 7, 6),
    "Friday": datetime(2017, 7, 7),
}
DAY_START, DAY_END = (8, 55), (17, 5)
ATTACK_WINDOWS = {
    "FTP-Patator": [("Tuesday", (9, 20), (10, 20))],
    "SSH-Patator": [("Tuesday", (14, 0), (15, 0))],
    "DoS slowloris": [("Wednesday", (9, 47), (10, 10))],
    "DoS Slowhttptest": [("Wednesday", (10, 14), (10, 35))],
    "DoS Hulk": [("Wednesday", (10, 43), (11, 0))],
    "DoS GoldenEye": [("Wednesday", (11, 10), (11, 23))],
    "Heartbleed": [("Wednesday", (15, 12), (15, 32))],
    "Web Attack - Brute Force": [("Thursday", (9, 20), (10, 0))],
    "Web Attack - XSS": [("Thursday", (10, 15), (10, 35))],
    "Web Attack - Sql Injection": [("Thursday", (10, 40), (10, 42))],
    "Infiltration": [("Thursday", (14, 19), (15, 45))],
    "Bot": [("Friday", (10, 2), (11, 2))],
    "PortScan": [("Friday", (13, 55), (15, 29))],
    "DDoS": [("Friday", (15, 56), (16, 16))],
}
# Share of benign traffic per day (Monday is benign-only and the largest file).
BENIGN_DAY_WEIGHTS = {"Monday": 0.25, "Tuesday": 0.19, "Wednesday": 0.20, "Thursday": 0.18, "Friday": 0.18}

# Testbed addresses published for CICIDS2017.
ATTACKER_EXT = "205.174.165.73"
ATTACKER_NAT = "172.16.0.1"
WEB_SERVER = "192.168.10.50"
UBUNTU_SERVER = "192.168.10.51"
INTERNAL_HOSTS = [f"192.168.10.{i}" for i in (3, 5, 8, 9, 12, 14, 15, 16, 17, 19, 25)]
BOT_VICTIMS = ["192.168.10.5", "192.168.10.8", "192.168.10.9", "192.168.10.14", "192.168.10.15"]

HARD_ATTACK_FRACTION = 0.04  # attack flows whose statistics look benign
BENIGN_OUTLIER_FRACTION = 0.003  # unusual-but-benign flows (drive IF false positives)
DUPLICATE_FRACTION = 0.01


# ---------------------------------------------------------------------------------
# Primitive samplers
# ---------------------------------------------------------------------------------
def _ln(rng, median, sigma, n):
    """Log-normal with a given median."""
    return rng.lognormal(np.log(median), sigma, n)


def _count(rng, median, sigma, n, lo=1):
    return np.maximum(lo, np.round(_ln(rng, median, sigma, n))).astype(np.int64)


def _choice(rng, values, n, p=None):
    return rng.choice(np.asarray(values), size=n, p=p)


def _base(rng, n, *, dport, proto=6, dur, nf, nb, fmean, bmean, fcv=0.6, bcv=0.6,
          iat_cv=1.2, flags=None, win_f=(29200,), win_b=(29200,), hdr=(20, 32),
          data_frac=(0.3, 0.7)):
    """Assemble a dict of primitives. Scalars are broadcast."""
    flags = flags or {}

    def arr(x, dtype=float):
        return np.broadcast_to(np.asarray(x, dtype=dtype), (n,)).copy()

    proto = arr(proto, np.int64)
    return {
        "dport": arr(dport, np.int64),
        "proto": proto,
        "dur": arr(dur),
        "nf": arr(nf, np.int64),
        "nb": arr(nb, np.int64),
        "fmean": arr(fmean),
        "bmean": arr(bmean),
        "fcv": arr(fcv),
        "bcv": arr(bcv),
        "iat_cv": arr(iat_cv),
        "p_fin": arr(flags.get("fin", 0.3)),
        "p_syn": arr(flags.get("syn", 0.05)),
        "p_rst": arr(flags.get("rst", 0.02)),
        "p_psh": arr(flags.get("psh", 0.35)),
        "p_ack": arr(flags.get("ack", 0.3)),
        "p_urg": arr(flags.get("urg", 0.05)),
        "p_ece": arr(flags.get("ece", 0.0)),
        "win_f": _choice(rng, win_f, n).astype(np.int64),
        "win_b": _choice(rng, win_b, n).astype(np.int64),
        "hdr": _choice(rng, hdr, n).astype(np.int64),
        "data_frac": rng.uniform(*data_frac, n),
    }


def _concat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _split(n, weights, rng):
    return rng.multinomial(n, np.asarray(weights) / np.sum(weights))


def benign(rng, n):
    https, http, dns, other, outlier = _split(
        n, [0.44, 0.15, 0.29, 0.11, BENIGN_OUTLIER_FRACTION], rng
    )
    parts = [
        _base(rng, https, dport=443, dur=_ln(rng, 1.5e6, 1.6, https),
              nf=_count(rng, 9, 0.8, https), nb=_count(rng, 10, 0.9, https, lo=0),
              fmean=_ln(rng, 110, 0.6, https), bmean=_ln(rng, 650, 0.7, https),
              iat_cv=_ln(rng, 1.6, 0.4, https),
              win_f=(8192, 29200, 65535, 64240), win_b=(-1, 28960, 65535, 4380, 5840)),
        _base(rng, http, dport=_choice(rng, [80, 8080, 80, 80], http),
              dur=_ln(rng, 8e5, 1.6, http),
              nf=_count(rng, 6, 0.7, http), nb=_count(rng, 6, 0.8, http, lo=0),
              fmean=_ln(rng, 160, 0.6, http), bmean=_ln(rng, 700, 0.7, http),
              iat_cv=_ln(rng, 1.4, 0.4, http),
              win_f=(8192, 29200, 65535), win_b=(-1, 28960, 65535, 5840)),
        _base(rng, dns, dport=_choice(rng, [53, 53, 53, 137, 123], dns), proto=17,
              dur=_ln(rng, 4e4, 1.8, dns),
              nf=_count(rng, 1.3, 0.4, dns), nb=_count(rng, 1.3, 0.4, dns, lo=0),
              fmean=_ln(rng, 38, 0.3, dns), bmean=_ln(rng, 110, 0.6, dns),
              fcv=0.1, bcv=0.2, iat_cv=0.3, flags={"fin": 0, "syn": 0, "rst": 0, "psh": 0, "ack": 0, "urg": 0},
              win_f=(-1,), win_b=(-1,), hdr=(8,), data_frac=(1.0, 1.0)),
        _base(rng, other, dport=_choice(rng, [22, 21, 139, 445, 389, 88, 3268, 5353]
                                        + list(rng.integers(1024, 65535, 8)), other),
              dur=_ln(rng, 3e6, 1.8, other),
              nf=_count(rng, 5, 0.9, other), nb=_count(rng, 4, 0.9, other, lo=0),
              fmean=_ln(rng, 90, 0.8, other), bmean=_ln(rng, 180, 0.9, other),
              iat_cv=_ln(rng, 1.5, 0.5, other),
              win_f=(8192, 29200, 65535, -1), win_b=(-1, 0, 28960, 65535)),
        # Heavy-tailed benign minority: long transfers, backups, streaming.
        _base(rng, outlier, dport=_choice(rng, [443, 80, 445, 22, 993], outlier),
              dur=_ln(rng, 6e7, 1.0, outlier),
              nf=_count(rng, 400, 1.2, outlier), nb=_count(rng, 700, 1.2, outlier),
              fmean=_ln(rng, 300, 1.0, outlier), bmean=_ln(rng, 1200, 0.5, outlier),
              iat_cv=_ln(rng, 3.0, 0.6, outlier),
              win_f=(8192, 29200, 65535), win_b=(28960, 65535)),
    ]
    return _concat([p for p in parts if len(p["dport"])])


def dos_hulk(rng, n):
    return _base(rng, n, dport=80, dur=_ln(rng, 1.2e4, 1.4, n),
                 nf=_count(rng, 5, 0.3, n), nb=_count(rng, 5, 0.4, n, lo=0),
                 fmean=_ln(rng, 340, 0.2, n), bmean=_ln(rng, 4800, 0.3, n),
                 fcv=1.3, bcv=1.1, iat_cv=_ln(rng, 1.2, 0.4, n),
                 flags={"fin": 0.6, "psh": 0.2, "ack": 0.55, "urg": 0.25},
                 win_f=(29200, 251, 256), win_b=(235, 227, -1))


def ddos_loic(rng, n):
    return _base(rng, n, dport=80, dur=_ln(rng, 1.6e6, 0.8, n),
                 nf=_count(rng, 4, 0.25, n), nb=_count(rng, 4, 0.3, n),
                 fmean=_ln(rng, 7, 0.3, n), bmean=_ln(rng, 2900, 0.2, n),
                 fcv=1.5, bcv=1.2, iat_cv=_ln(rng, 1.3, 0.3, n),
                 flags={"fin": 0.1, "psh": 0.05, "ack": 0.9, "urg": 0.0},
                 win_f=(8192, 256), win_b=(229, -1), data_frac=(0.2, 0.4))


def dos_goldeneye(rng, n):
    return _base(rng, n, dport=80, dur=_ln(rng, 1.1e7, 0.3, n),
                 nf=_count(rng, 10, 0.2, n), nb=_count(rng, 8, 0.2, n, lo=0),
                 fmean=_ln(rng, 420, 0.2, n), bmean=_ln(rng, 3200, 0.3, n),
                 fcv=1.6, bcv=1.4, iat_cv=_ln(rng, 3.5, 0.2, n),
                 flags={"fin": 0.2, "psh": 0.6, "ack": 0.3},
                 win_f=(29200,), win_b=(235,))


def dos_slowloris(rng, n):
    return _base(rng, n, dport=80, dur=_ln(rng, 9e7, 0.3, n),
                 nf=_count(rng, 6, 0.4, n), nb=_count(rng, 3, 0.5, n, lo=0),
                 fmean=_ln(rng, 30, 0.3, n), bmean=np.zeros(n),
                 iat_cv=_ln(rng, 0.7, 0.3, n),
                 flags={"fin": 0.05, "psh": 0.8, "ack": 0.2},
                 win_f=(29200,), win_b=(28960, 235))


def dos_slowhttptest(rng, n):
    return _base(rng, n, dport=80, dur=_ln(rng, 1.1e8, 0.3, n),
                 nf=_count(rng, 5, 0.4, n), nb=_count(rng, 2, 0.5, n, lo=0),
                 fmean=_ln(rng, 22, 0.4, n), bmean=np.zeros(n),
                 iat_cv=_ln(rng, 0.9, 0.3, n),
                 flags={"fin": 0.1, "psh": 0.7, "ack": 0.3},
                 win_f=(29200,), win_b=(28960,))


def portscan(rng, n):
    common = [21, 22, 23, 25, 53, 80, 110, 111, 135, 139, 143, 443, 445, 993, 995,
              1723, 3306, 3389, 5900, 8080]
    ports = np.where(rng.random(n) < 0.35, _choice(rng, common, n), rng.integers(1, 65535, n))
    single = rng.random(n) < 0.85
    dur = np.where(single, np.where(rng.random(n) < 0.03, 0.0, _ln(rng, 40, 1.0, n)),
                   _ln(rng, 80, 1.2, n))
    return _base(rng, n, dport=ports, dur=dur,
                 nf=np.where(single, 1, 2), nb=np.where(rng.random(n) < 0.8, 1, 0),
                 fmean=np.zeros(n), bmean=np.zeros(n), fcv=0, bcv=0, iat_cv=0.2,
                 flags={"fin": 0.0, "syn": 0.9, "rst": 0.7, "psh": 0.02, "ack": 0.1, "urg": 0.0},
                 win_f=(1024, 29200), win_b=(0, -1), hdr=(24, 20), data_frac=(0, 0))


def ftp_patator(rng, n):
    return _base(rng, n, dport=21, dur=_ln(rng, 4.5e6, 0.5, n),
                 nf=_count(rng, 10, 0.2, n), nb=_count(rng, 15, 0.2, n),
                 fmean=_ln(rng, 9, 0.2, n), bmean=_ln(rng, 24, 0.2, n),
                 fcv=0.8, bcv=0.5, iat_cv=_ln(rng, 1.5, 0.2, n),
                 flags={"fin": 0.9, "psh": 0.9, "ack": 0.1},
                 win_f=(29200,), win_b=(227,), hdr=(32,))


def ssh_patator(rng, n):
    return _base(rng, n, dport=22, dur=_ln(rng, 1.2e7, 0.2, n),
                 nf=_count(rng, 30, 0.1, n), nb=_count(rng, 45, 0.1, n),
                 fmean=_ln(rng, 45, 0.1, n), bmean=_ln(rng, 70, 0.1, n),
                 fcv=1.7, bcv=2.0, iat_cv=_ln(rng, 2.4, 0.15, n),
                 flags={"fin": 0.9, "psh": 0.95, "ack": 0.1},
                 win_f=(29200,), win_b=(247,), hdr=(32,))


def bot_ares(rng, n):
    return _base(rng, n, dport=8080, dur=_ln(rng, 6e7, 0.15, n),
                 nf=_count(rng, 12, 0.15, n), nb=_count(rng, 10, 0.15, n),
                 fmean=_ln(rng, 190, 0.05, n), bmean=_ln(rng, 60, 0.05, n),
                 fcv=0.02, bcv=0.02, iat_cv=_ln(rng, 0.08, 0.3, n),
                 flags={"fin": 0.4, "psh": 0.5, "ack": 0.5},
                 win_f=(8192,), win_b=(29200, 237))


def web_bruteforce(rng, n):
    return _base(rng, n, dport=80, dur=_ln(rng, 5.6e6, 0.3, n),
                 nf=_count(rng, 3, 0.3, n), nb=_count(rng, 1, 0.3, n, lo=0),
                 fmean=_ln(rng, 290, 0.15, n), bmean=_ln(rng, 18, 0.5, n),
                 fcv=0.05, iat_cv=_ln(rng, 3.0, 0.15, n),
                 flags={"fin": 0.2, "psh": 0.9, "ack": 0.4},
                 win_f=(29200,), win_b=(28960,))


def web_xss(rng, n):
    return _base(rng, n, dport=80, dur=_ln(rng, 5.3e6, 0.3, n),
                 nf=_count(rng, 5, 0.3, n), nb=_count(rng, 4, 0.3, n, lo=0),
                 fmean=_ln(rng, 680, 0.15, n), bmean=_ln(rng, 2600, 0.2, n),
                 fcv=1.8, iat_cv=_ln(rng, 3.0, 0.15, n),
                 flags={"fin": 0.3, "psh": 0.9, "ack": 0.4},
                 win_f=(29200,), win_b=(28960,))


def web_sqli(rng, n):
    return _base(rng, n, dport=80, dur=_ln(rng, 5e6, 0.35, n),
                 nf=_count(rng, 5, 0.3, n), nb=_count(rng, 4, 0.3, n, lo=0),
                 fmean=_ln(rng, 560, 0.15, n), bmean=_ln(rng, 3400, 0.2, n),
                 fcv=1.8, iat_cv=_ln(rng, 3.0, 0.15, n),
                 flags={"fin": 0.3, "psh": 0.9, "ack": 0.4},
                 win_f=(29200,), win_b=(28960,))


def infiltration(rng, n):
    return _base(rng, n, dport=_choice(rng, [444, 444, 4444, 445, 139], n),
                 dur=_ln(rng, 3e7, 1.5, n),
                 nf=_count(rng, 60, 0.8, n), nb=_count(rng, 90, 0.8, n),
                 fmean=_ln(rng, 220, 0.5, n), bmean=_ln(rng, 1400, 0.4, n),
                 iat_cv=_ln(rng, 2.5, 0.4, n),
                 flags={"fin": 0.3, "psh": 0.7, "ack": 0.5, "urg": 0.1},
                 win_f=(8192, 65535), win_b=(8192, 65535))


def heartbleed(rng, n):
    return _base(rng, n, dport=444, dur=_ln(rng, 1.15e8, 0.1, n),
                 nf=_count(rng, 2700, 0.15, n), nb=_count(rng, 1900, 0.15, n),
                 fmean=_ln(rng, 20, 0.1, n), bmean=_ln(rng, 4000, 0.08, n),
                 fcv=2.0, bcv=0.4, iat_cv=_ln(rng, 2.0, 0.2, n),
                 flags={"fin": 0.1, "psh": 0.9, "ack": 0.2},
                 win_f=(29200,), win_b=(235,))


GENERATORS = {
    "BENIGN": benign,
    "DoS Hulk": dos_hulk,
    "DDoS": ddos_loic,
    "DoS GoldenEye": dos_goldeneye,
    "DoS slowloris": dos_slowloris,
    "DoS Slowhttptest": dos_slowhttptest,
    "PortScan": portscan,
    "FTP-Patator": ftp_patator,
    "SSH-Patator": ssh_patator,
    "Bot": bot_ares,
    "Web Attack - Brute Force": web_bruteforce,
    "Web Attack - XSS": web_xss,
    "Web Attack - Sql Injection": web_sqli,
    "Infiltration": infiltration,
    "Heartbleed": heartbleed,
}


def _hardify(rng, prim, frac):
    """Replace the statistical primitives of a fraction of attack flows with benign-web
    ones (keeping port/protocol), mimicking attack-window flows that look benign."""
    n = len(prim["dport"])
    mask = rng.random(n) < frac
    k = int(mask.sum())
    if k == 0:
        return prim
    b = benign(rng, k)
    for key in ("dur", "nf", "nb", "fmean", "bmean", "fcv", "bcv", "iat_cv",
                "p_fin", "p_syn", "p_rst", "p_psh", "p_ack", "p_urg", "win_f", "win_b"):
        prim[key][mask] = b[key][:k]
    prim["proto"][mask] = 6
    return prim


# ---------------------------------------------------------------------------------
# CICFlowMeter-style feature derivation
# ---------------------------------------------------------------------------------
def derive_features(prim: dict, rng) -> pd.DataFrame:
    n = len(prim["dport"])
    nf = prim["nf"].astype(float)
    nb = prim["nb"].astype(float)
    ntot = nf + nb
    dur = np.round(prim["dur"]).astype(float)
    dur[ntot <= 1] = np.where(rng.random(int((ntot <= 1).sum())) < 0.2, 0.0, dur[ntot <= 1])

    def lengths(count, mean, cv):
        mean = np.where(count > 0, mean, 0.0)
        std = np.where(count > 1, mean * cv * rng.uniform(0.6, 1.4, n), 0.0)
        mx = np.where(count > 1, mean + std * rng.uniform(1.0, 2.5, n), mean)
        mn = np.where(count > 1, np.maximum(0.0, mean - std * rng.uniform(0.5, 1.5, n)), mean)
        tot = np.round(count * mean)
        return np.round(mean, 4), np.round(std, 4), np.round(mx), np.round(mn), tot

    fmean, fstd, fmax, fmin, ftot = lengths(nf, prim["fmean"], prim["fcv"])
    bmean, bstd, bmax, bmin, btot = lengths(nb, prim["bmean"], prim["bcv"])

    with np.errstate(divide="ignore", invalid="ignore"):
        flow_bytes_s = (ftot + btot) / dur * 1e6
        flow_pkts_s = ntot / dur * 1e6
        fwd_pkts_s = nf / dur * 1e6
        bwd_pkts_s = nb / dur * 1e6
    # CICFlowMeter: packets/s columns are finite (0 duration -> huge value) in the
    # per-direction columns, but NaN/Infinity in the Flow-level ones.
    fwd_pkts_s = np.where(np.isfinite(fwd_pkts_s), fwd_pkts_s, 2e6)
    bwd_pkts_s = np.where(np.isfinite(bwd_pkts_s), bwd_pkts_s, 0.0)

    def iat(count, total, cv):
        gaps = np.maximum(count - 1, 0)
        mean = np.where(gaps > 0, total / np.maximum(gaps, 1), 0.0)
        std = np.where(gaps > 1, mean * cv, 0.0)
        mx = np.where(gaps > 0, np.minimum(total, mean + std * rng.uniform(1.0, 3.0, n)), 0.0)
        mx = np.maximum(mx, mean)
        mn = np.where(gaps > 0, mean * rng.uniform(0.0, 0.3, n), 0.0)
        return np.round(total * (gaps > 0)), mean, std, np.round(mx), np.round(mn)

    _, flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = iat(ntot, dur, prim["iat_cv"])
    fwd_tot = dur * rng.uniform(0.6, 1.0, n)
    bwd_tot = dur * rng.uniform(0.5, 1.0, n)
    f_iat = iat(nf, fwd_tot, prim["iat_cv"] * rng.uniform(0.8, 1.2, n))
    b_iat = iat(nb, bwd_tot, prim["iat_cv"] * rng.uniform(0.8, 1.2, n))

    tcp = prim["proto"] == 6
    hdr = np.where(tcp, prim["hdr"], 8)
    fwd_hdr = nf * hdr
    bwd_hdr = nb * hdr

    min_len = np.where(nb > 0, np.minimum(fmin, bmin), fmin)
    max_len = np.where(nb > 0, np.maximum(fmax, bmax), fmax)
    pkt_mean = np.where(ntot > 0, (ftot + btot) / (ntot + 1), 0.0)
    pooled_var = np.where(ntot > 1, (nf * (fstd**2 + (fmean - pkt_mean) ** 2)
                                     + nb * (bstd**2 + (bmean - pkt_mean) ** 2)) / ntot, 0.0)
    pkt_std = np.sqrt(pooled_var)

    def flag(p):
        return (rng.random(n) < p).astype(np.int64) * tcp

    fin, syn, rst = flag(prim["p_fin"]), flag(prim["p_syn"]), flag(prim["p_rst"])
    psh, ack, urg = flag(prim["p_psh"]), flag(prim["p_ack"]), flag(prim["p_urg"])
    ece = flag(prim["p_ece"])
    fwd_psh = np.where(psh == 1, (rng.random(n) < 0.3).astype(np.int64), 0)

    win_f = np.where(tcp, prim["win_f"], -1)
    win_b = np.where(tcp & (nb > 0), prim["win_b"], -1)
    act_data = np.where(fmean > 0, np.maximum(1, np.round(nf * prim["data_frac"])), 0)
    min_seg = np.where(tcp, np.where(prim["hdr"] >= 32, 32, 20), 8) * (rng.random(n) > 0.02)

    long_flow = (dur > 5e6) & (ntot > 2)
    active_mean = np.where(long_flow, np.minimum(dur * 0.2, _ln(rng, 2e5, 1.5, n)), 0.0)
    active_std = np.where(long_flow & (rng.random(n) < 0.4), active_mean * rng.uniform(0.1, 0.8, n), 0.0)
    active_max = active_mean + active_std * rng.uniform(1.0, 2.0, n)
    active_min = np.maximum(0.0, active_mean - active_std * rng.uniform(0.5, 1.2, n))
    idle_mean = np.where(long_flow, dur * rng.uniform(0.3, 0.9, n), 0.0)
    idle_std = np.where(long_flow & (rng.random(n) < 0.3), idle_mean * rng.uniform(0.05, 0.5, n), 0.0)
    idle_max = np.minimum(dur, idle_mean + idle_std * rng.uniform(1.0, 2.0, n))
    idle_max = np.maximum(idle_max, idle_mean)
    idle_min = np.maximum(0.0, idle_mean - idle_std * rng.uniform(0.5, 1.2, n))

    zeros = np.zeros(n, dtype=np.int64)
    cols = {
        "Destination Port": prim["dport"],
        "Flow Duration": dur,
        "Total Fwd Packets": nf,
        "Total Backward Packets": nb,
        "Total Length of Fwd Packets": ftot,
        "Total Length of Bwd Packets": btot,
        "Fwd Packet Length Max": fmax,
        "Fwd Packet Length Min": fmin,
        "Fwd Packet Length Mean": fmean,
        "Fwd Packet Length Std": fstd,
        "Bwd Packet Length Max": bmax,
        "Bwd Packet Length Min": bmin,
        "Bwd Packet Length Mean": bmean,
        "Bwd Packet Length Std": bstd,
        "Flow Bytes/s": flow_bytes_s,
        "Flow Packets/s": flow_pkts_s,
        "Flow IAT Mean": flow_iat_mean,
        "Flow IAT Std": flow_iat_std,
        "Flow IAT Max": flow_iat_max,
        "Flow IAT Min": flow_iat_min,
        "Fwd IAT Total": f_iat[0],
        "Fwd IAT Mean": f_iat[1],
        "Fwd IAT Std": f_iat[2],
        "Fwd IAT Max": f_iat[3],
        "Fwd IAT Min": f_iat[4],
        "Bwd IAT Total": b_iat[0],
        "Bwd IAT Mean": b_iat[1],
        "Bwd IAT Std": b_iat[2],
        "Bwd IAT Max": b_iat[3],
        "Bwd IAT Min": b_iat[4],
        "Fwd PSH Flags": fwd_psh,
        "Bwd PSH Flags": zeros,
        "Fwd URG Flags": zeros,
        "Bwd URG Flags": zeros,
        "Fwd Header Length": fwd_hdr,
        "Bwd Header Length": bwd_hdr,
        "Fwd Packets/s": fwd_pkts_s,
        "Bwd Packets/s": bwd_pkts_s,
        "Min Packet Length": min_len,
        "Max Packet Length": max_len,
        "Packet Length Mean": pkt_mean,
        "Packet Length Std": pkt_std,
        "Packet Length Variance": pooled_var,
        "FIN Flag Count": fin,
        "SYN Flag Count": syn,
        "RST Flag Count": rst,
        "PSH Flag Count": psh,
        "ACK Flag Count": ack,
        "URG Flag Count": urg,
        "CWE Flag Count": zeros,
        "ECE Flag Count": ece,
        "Down/Up Ratio": np.floor(nb / np.maximum(nf, 1)),
        "Average Packet Size": np.where(ntot > 0, pkt_mean * (ntot + 1) / ntot, 0.0),
        "Avg Fwd Segment Size": fmean,
        "Avg Bwd Segment Size": bmean,
        "Fwd Header Length.1": fwd_hdr,
        "Fwd Avg Bytes/Bulk": zeros,
        "Fwd Avg Packets/Bulk": zeros,
        "Fwd Avg Bulk Rate": zeros,
        "Bwd Avg Bytes/Bulk": zeros,
        "Bwd Avg Packets/Bulk": zeros,
        "Bwd Avg Bulk Rate": zeros,
        "Subflow Fwd Packets": nf,
        "Subflow Fwd Bytes": ftot,
        "Subflow Bwd Packets": nb,
        "Subflow Bwd Bytes": btot,
        "Init_Win_bytes_forward": win_f,
        "Init_Win_bytes_backward": win_b,
        "act_data_pkt_fwd": act_data,
        "min_seg_size_forward": min_seg,
        "Active Mean": active_mean,
        "Active Std": active_std,
        "Active Max": active_max,
        "Active Min": active_min,
        "Idle Mean": idle_mean,
        "Idle Std": idle_std,
        "Idle Max": idle_max,
        "Idle Min": idle_min,
    }
    df = pd.DataFrame(cols, columns=FEATURE_COLUMNS)
    int_like = [c for c in FEATURE_COLUMNS if c not in (
        "Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s",
        "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Mean",
        "Bwd Packet Length Std", "Flow IAT Mean", "Flow IAT Std", "Fwd IAT Mean",
        "Fwd IAT Std", "Bwd IAT Mean", "Bwd IAT Std", "Packet Length Mean",
        "Packet Length Std", "Packet Length Variance", "Average Packet Size",
        "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Active Mean", "Active Std",
        "Idle Mean", "Idle Std")]
    df[int_like] = df[int_like].round().astype(np.int64)
    return df


# ---------------------------------------------------------------------------------
# Identifiers, timestamps, labels
# ---------------------------------------------------------------------------------
def _times_in(rng, day, start, end, n):
    base = DAYS[day]
    t0 = base.replace(hour=start[0], minute=start[1])
    t1 = base.replace(hour=end[0], minute=end[1])
    secs = rng.uniform(0, (t1 - t0).total_seconds(), n)
    return [t0 + timedelta(seconds=float(s)) for s in secs]


def _timestamps(rng, label, n):
    if label == "BENIGN":
        days = _choice(rng, list(BENIGN_DAY_WEIGHTS), n, p=list(BENIGN_DAY_WEIGHTS.values()))
        out = np.empty(n, dtype=object)
        for day in DAYS:
            idx = np.where(days == day)[0]
            out[idx] = _times_in(rng, day, DAY_START, DAY_END, len(idx))
        return list(out)
    windows = ATTACK_WINDOWS[label]
    w = _choice(rng, np.arange(len(windows)), n)
    out = np.empty(n, dtype=object)
    for i, (day, s, e) in enumerate(windows):
        idx = np.where(w == i)[0]
        out[idx] = _times_in(rng, day, s, e, len(idx))
    return list(out)


def _format_ts(ts: datetime) -> str:
    # CICIDS2017 style: day/month/year, no zero padding, 12-hour clock *without* AM/PM
    # (e.g. 15:56 is written "7/7/2017 3:56"). prepare_cicids2017 repairs this.
    hour = ts.hour % 12 or 12
    return f"{ts.day}/{ts.month}/{ts.year} {hour}:{ts.minute:02d}:{ts.second:02d}"


def _random_public_ips(rng, n):
    first = _choice(rng, [8, 13, 23, 31, 52, 91, 104, 151, 172, 185, 199, 216], n)
    return [f"{a}.{b}.{c}.{d}" for a, b, c, d in zip(
        first, rng.integers(0, 255, n), rng.integers(0, 255, n), rng.integers(1, 254, n))]


def _addresses(rng, label, n):
    sport = rng.integers(1024, 65535, n)
    if label == "BENIGN":
        src = list(_choice(rng, INTERNAL_HOSTS, n))
        dst = _random_public_ips(rng, n)
        internal = rng.random(n) < 0.15
        for i in np.where(internal)[0]:
            dst[i] = f"192.168.10.{rng.integers(1, 60)}"
        return src, dst, sport
    if label == "Bot":
        return list(_choice(rng, BOT_VICTIMS, n)), [ATTACKER_EXT] * n, sport
    if label == "Infiltration":
        return ["192.168.10.8"] * n, [ATTACKER_EXT] * n, sport
    if label == "Heartbleed":
        return [ATTACKER_NAT] * n, [UBUNTU_SERVER] * n, sport
    if label in ("FTP-Patator", "SSH-Patator"):
        return [ATTACKER_NAT] * n, [UBUNTU_SERVER] * n, sport
    return [ATTACKER_NAT] * n, [WEB_SERVER] * n, sport


def class_counts(total: int, min_per_class: int) -> dict[str, int]:
    real_total = sum(REAL_LABEL_COUNTS.values())
    counts = {k: max(min_per_class, int(round(total * v / real_total)))
              for k, v in REAL_LABEL_COUNTS.items()}
    counts["BENIGN"] -= sum(counts.values()) - total  # keep the grand total exact
    return counts


def generate(total: int = 120_000, seed: int = 42, min_per_class: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for label, n in class_counts(total, min_per_class).items():
        prim = GENERATORS[label](rng, n)
        if label != "BENIGN":
            prim = _hardify(rng, prim, HARD_ATTACK_FRACTION)
        feats = derive_features(prim, rng)
        src, dst, sport = _addresses(rng, label, n)
        ts = _timestamps(rng, label, n)
        proto = prim["proto"]
        ids = pd.DataFrame({
            "Flow ID": [f"{d}-{s}-{dp}-{sp}-{p}" for s, d, sp, dp, p in
                        zip(src, dst, sport, feats["Destination Port"], proto)],
            "Source IP": src,
            "Source Port": sport,
            "Destination IP": dst,
            "Protocol": proto,
            "_ts": ts,
        })
        df = pd.concat([ids, feats], axis=1)
        df["Label"] = RAW_LABEL_SPELLING.get(label, label)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    dup = data.sample(frac=DUPLICATE_FRACTION, random_state=seed)
    data = pd.concat([data, dup], ignore_index=True)
    data = data.sort_values("_ts", kind="stable").reset_index(drop=True)
    data["Timestamp"] = [_format_ts(t) for t in data["_ts"]]
    data["_day"] = [t.strftime("%A") for t in data["_ts"]]
    return data


def to_raw_csv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Order columns as in the raw CICIDS2017 files and reproduce their header quirks."""
    out = df[RAW_COLUMN_ORDER].copy()
    for c in ("Flow Bytes/s", "Flow Packets/s"):
        s = out[c].astype(object)
        s[np.isposinf(out[c].to_numpy())] = "Infinity"
        s[np.isnan(out[c].to_numpy(dtype=float))] = "NaN"
        out[c] = s
    headers = ["Flow ID"] + [" " + c for c in RAW_COLUMN_ORDER[1:]]
    headers = [" Fwd Header Length" if h.strip() == "Fwd Header Length.1" else h for h in headers]
    out.columns = headers
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=120_000, help="flows before duplicates (default 120000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-per-class", type=int, default=60,
                    help="floor for the rarest labels (real CICIDS2017 has 11 Heartbleed flows)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "synthetic" / "synthetic_cicids2017.csv")
    ap.add_argument("--per-day", action="store_true",
                    help="also write one CSV per weekday, like the real dataset's file layout")
    args = ap.parse_args(argv)

    print(SYNTHETIC_BANNER)
    df = generate(args.rows, args.seed, args.min_per_class)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw = to_raw_csv_frame(df)
    raw.to_csv(args.out, index=False, float_format="%.6g")

    if args.per_day:
        for day, idx in df.groupby("_day").groups.items():
            p = args.out.with_name(f"{args.out.stem}_{day}.csv")
            raw.loc[idx].to_csv(p, index=False, float_format="%.6g")
            print(f"  wrote {p.name}: {len(idx):,} rows")

    counts = df["Label"].value_counts()
    meta = {
        "data_source": "synthetic",
        "generator": "training/generate_synthetic_flows.py",
        "seed": args.seed,
        "rows": int(len(df)),
        "duplicates_injected": int(round(args.rows * DUPLICATE_FRACTION)),
        "label_counts": {k: int(v) for k, v in counts.items()},
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "warning": "SYNTHETIC DATA - metrics computed on this file are not real evaluation results.",
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"Wrote {len(df):,} flows x {raw.shape[1]} columns -> {args.out}")
    print(f"Benign share: {(df['Label'] == 'BENIGN').mean():.1%}")
    print(counts.to_string())
    print(SYNTHETIC_BANNER)


if __name__ == "__main__":
    main()
