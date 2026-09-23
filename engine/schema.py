"""CICIDS2017 column schema and threat-category mapping.

Single source of truth shared by the synthetic generator, the dataset preparation
script, the ingestion layer and the dashboard.

Column names follow the CICIDS2017 "GeneratedLabelledFlows" / "TrafficLabelling" CSVs
(the CICFlowMeter-V3 output with identifier columns kept). The raw CSV headers carry
leading spaces (e.g. " Destination Port") and one duplicated column
("Fwd Header Length.1"); ``normalise_columns`` strips the whitespace so both the real
files and the synthetic file resolve to the canonical names below.
"""
from __future__ import annotations

import re

# --- Identifier / time columns ------------------------------------------------------
# Dropped from the model input but carried alongside every flow for logging.
ID_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Protocol",
    "Timestamp",
]

LABEL_COLUMN = "Label"

# --- The 78 CICFlowMeter feature columns (MachineLearningCSV order) -------------------
FEATURE_COLUMNS = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]
assert len(FEATURE_COLUMNS) == 78, len(FEATURE_COLUMNS)

# Full column order of a CICIDS2017 TrafficLabelling CSV (84 columns + Label).
# "Destination Port" sits between "Destination IP" and "Protocol" in the raw files.
RAW_COLUMN_ORDER = (
    ["Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol", "Timestamp"]
    + [c for c in FEATURE_COLUMNS if c != "Destination Port"]
    + [LABEL_COLUMN]
)

# Heavy-tailed count / byte / time features that get a log1p transform.
LOG1P_PATTERNS = (
    "Duration", "Packets", "Length", "Bytes", "IAT", "/s", "Header",
    "Size", "Subflow", "Active", "Idle", "Variance", "Init_Win", "act_data",
)


def is_heavy_tailed(column: str) -> bool:
    return any(p in column for p in LOG1P_PATTERNS)


# --- Threat categories ----------------------------------------------------------------
BENIGN = "Benign"
CAT_DOS = "DoS/DDoS"
CAT_PORTSCAN = "Port Scan"
CAT_BRUTE = "Brute Force"
CAT_WEB = "Web Attack"
CAT_BOT = "Botnet (Malware)"
CAT_OTHER = "Other (Infiltration/Heartbleed)"
CAT_PHISHING = "Phishing-related"  # reserved: CICIDS2017 has no phishing flows
UNKNOWN_ANOMALY = "Unknown anomaly"

THREAT_CATEGORIES = [BENIGN, CAT_DOS, CAT_PORTSCAN, CAT_BRUTE, CAT_WEB, CAT_BOT, CAT_OTHER, CAT_PHISHING]
TRAINED_CATEGORIES = [c for c in THREAT_CATEGORIES if c != CAT_PHISHING]

# Canonical CICIDS2017 labels -> threat category.
LABEL_TO_CATEGORY = {
    "BENIGN": BENIGN,
    "DoS Hulk": CAT_DOS,
    "DDoS": CAT_DOS,
    "DoS GoldenEye": CAT_DOS,
    "DoS slowloris": CAT_DOS,
    "DoS Slowhttptest": CAT_DOS,
    "PortScan": CAT_PORTSCAN,
    "FTP-Patator": CAT_BRUTE,
    "SSH-Patator": CAT_BRUTE,
    "Web Attack - Brute Force": CAT_WEB,
    "Web Attack - XSS": CAT_WEB,
    "Web Attack - Sql Injection": CAT_WEB,
    "Bot": CAT_BOT,
    "Infiltration": CAT_OTHER,
    "Heartbleed": CAT_OTHER,
}
CICIDS_LABELS = list(LABEL_TO_CATEGORY)

# Attack family used by evaluation Protocol C (leave-one-attack-family-out).
CATEGORY_FAMILY = {c: c for c in TRAINED_CATEGORIES if c != BENIGN}


def canonical_label(raw: str) -> str:
    """Normalise a raw CICIDS2017 label.

    The official CSVs contain the web-attack labels with a mangled en-dash
    ("Web Attack � Brute Force", "Web Attack \x96 XSS", ...) depending on how the file
    was decoded; this maps all variants onto the canonical ASCII spelling.
    """
    s = str(raw).strip()
    m = re.match(r"(?i)^web attack\W+(.*)$", s)
    if m:
        tail = m.group(1).strip().lower()
        if "brute" in tail:
            return "Web Attack - Brute Force"
        if "xss" in tail:
            return "Web Attack - XSS"
        if "sql" in tail:
            return "Web Attack - Sql Injection"
    for known in CICIDS_LABELS:
        if s.lower() == known.lower():
            return known
    return s


def label_to_category(raw: str) -> str:
    return LABEL_TO_CATEGORY.get(canonical_label(raw), CAT_OTHER)


def normalise_columns(columns) -> list[str]:
    """Strip whitespace from raw CICIDS2017 headers (" Destination Port" -> "Destination Port")."""
    return [str(c).strip() for c in columns]


# --- Default severity per category (used by the policy engine & dashboard) -------------
CATEGORY_SEVERITY = {
    CAT_DOS: "High",
    CAT_PORTSCAN: "Medium",
    CAT_BRUTE: "High",
    CAT_WEB: "High",
    CAT_BOT: "Critical",
    CAT_OTHER: "High",
    CAT_PHISHING: "Medium",
    UNKNOWN_ANOMALY: "Medium-High",
}
