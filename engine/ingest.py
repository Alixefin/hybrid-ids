"""Layer 1 - Traffic ingestion.

Reads CICFlowMeter CSV output (CICIDS2017 "TrafficLabelling" / "MachineLearningCVE"
layouts, the synthetic stand-in, or CICFlowMeter-V4 output with its abbreviated column
names) into a ``FlowBatch``: identifier columns kept for logging, numeric feature
columns for the models, and the ground-truth label when the file carries one.

The fitted pre-processing pipeline (engine.preprocess) is applied afterwards by the
detector; this module only normalises the *shape* of the input.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from engine.schema import (
    FEATURE_COLUMNS,
    ID_COLUMNS,
    LABEL_COLUMN,
    canonical_label,
    label_to_category,
    normalise_columns,
)

# CICFlowMeter-V4 (CSE-CIC-IDS2018 style) abbreviations -> CICIDS2017 names.
V4_ALIASES = {
    "Src IP": "Source IP", "Dst IP": "Destination IP", "Src Port": "Source Port",
    "Dst Port": "Destination Port", "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets", "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets", "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd Pkt Len Min": "Fwd Packet Length Min", "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Fwd Pkt Len Std": "Fwd Packet Length Std", "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Pkt Len Min": "Bwd Packet Length Min", "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Bwd Pkt Len Std": "Bwd Packet Length Std", "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s", "Fwd IAT Tot": "Fwd IAT Total", "Bwd IAT Tot": "Bwd IAT Total",
    "Fwd Header Len": "Fwd Header Length", "Bwd Header Len": "Bwd Header Length",
    "Fwd Pkts/s": "Fwd Packets/s", "Bwd Pkts/s": "Bwd Packets/s", "Pkt Len Min": "Min Packet Length",
    "Pkt Len Max": "Max Packet Length", "Pkt Len Mean": "Packet Length Mean",
    "Pkt Len Std": "Packet Length Std", "Pkt Len Var": "Packet Length Variance",
    "FIN Flag Cnt": "FIN Flag Count", "SYN Flag Cnt": "SYN Flag Count", "RST Flag Cnt": "RST Flag Count",
    "PSH Flag Cnt": "PSH Flag Count", "ACK Flag Cnt": "ACK Flag Count", "URG Flag Cnt": "URG Flag Count",
    "CWE Flag Cnt": "CWE Flag Count", "ECE Flag Cnt": "ECE Flag Count", "Pkt Size Avg": "Average Packet Size",
    "Fwd Seg Size Avg": "Avg Fwd Segment Size", "Bwd Seg Size Avg": "Avg Bwd Segment Size",
    "Fwd Byts/b Avg": "Fwd Avg Bytes/Bulk", "Fwd Pkts/b Avg": "Fwd Avg Packets/Bulk",
    "Fwd Blk Rate Avg": "Fwd Avg Bulk Rate", "Bwd Byts/b Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Pkts/b Avg": "Bwd Avg Packets/Bulk", "Bwd Blk Rate Avg": "Bwd Avg Bulk Rate",
    "Subflow Fwd Pkts": "Subflow Fwd Packets", "Subflow Fwd Byts": "Subflow Fwd Bytes",
    "Subflow Bwd Pkts": "Subflow Bwd Packets", "Subflow Bwd Byts": "Subflow Bwd Bytes",
    "Init Fwd Win Byts": "Init_Win_bytes_forward", "Init Bwd Win Byts": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd", "Fwd Seg Size Min": "min_seg_size_forward",
}

_TS_RE = re.compile(
    r"^\s*(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([AaPp][Mm])?\s*$"
)


@dataclass
class FlowBatch:
    ids: pd.DataFrame
    features: pd.DataFrame
    labels: pd.Series | None = None  # canonical CICIDS2017 label
    categories: pd.Series | None = None  # threat category
    source_name: str = ""
    rows_read: int = 0
    rows_dropped_empty: int = 0
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.features)

    @property
    def has_labels(self) -> bool:
        return self.labels is not None


def read_csv_any_encoding(src, **kwargs) -> pd.DataFrame:
    """Read a CSV as UTF-8, falling back to latin-1 (the official CICIDS2017 web-attack
    file contains a raw cp1252 0x96 dash that is not valid UTF-8)."""
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    try:
        return pd.read_csv(src, low_memory=False, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        if hasattr(src, "seek"):
            src.seek(0)
        return pd.read_csv(src, low_memory=False, encoding="latin-1", **kwargs)


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip header whitespace, map CICFlowMeter-V4 aliases, restore the duplicated
    'Fwd Header Length.1' column if the source lacks it."""
    df = df.copy()
    df.columns = normalise_columns(df.columns)
    df = df.rename(columns={k: v for k, v in V4_ALIASES.items() if k in df.columns})
    df = df.loc[:, ~df.columns.duplicated()]
    if "Fwd Header Length.1" not in df.columns and "Fwd Header Length" in df.columns:
        df["Fwd Header Length.1"] = df["Fwd Header Length"]
    return df


def parse_timestamps(values: pd.Series, capture_start_hour: int = 8) -> pd.Series:
    """Parse CICIDS2017 timestamps.

    The official files write "d/m/yyyy H:MM[:SS]" on a 12-hour clock *without* AM/PM
    (15:56 appears as "3:56"). Captures ran 08:xx-17:xx, so hours below
    ``capture_start_hour`` are afternoon hours. Anything else falls back to pandas.
    """
    s = values.astype(str)
    parts = s.str.extract(_TS_RE)
    ok = parts[0].notna()
    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if ok.any():
        p = parts[ok]
        hour = p[3].astype(int)
        ampm = p[6].fillna("").str.upper()
        hour = np.where(ampm == "PM", np.where(hour % 12 == 0, 12, hour % 12 + 12),
                        np.where(ampm == "AM", hour % 12,
                                 np.where(hour < capture_start_hour, hour + 12, hour)))
        out.loc[ok] = pd.to_datetime(pd.DataFrame({
            "year": p[2].astype(int), "month": p[1].astype(int), "day": p[0].astype(int),
            "hour": hour, "minute": p[4].astype(int), "second": p[5].fillna(0).astype(int),
        }), errors="coerce")
    if (~ok).any():
        out.loc[~ok] = pd.to_datetime(s[~ok], errors="coerce", dayfirst=True)
    return out


def coerce_features(df: pd.DataFrame, columns=None, dtype=np.float64) -> pd.DataFrame:
    """Numeric conversion of feature columns ('Infinity' / 'NaN' strings handled)."""
    cols = [c for c in (columns or FEATURE_COLUMNS) if c in df.columns]
    out = df[cols].apply(pd.to_numeric, errors="coerce")
    return out.astype(dtype)


def split_frame(df: pd.DataFrame, source_name: str = "", float_dtype=np.float64) -> FlowBatch:
    """Split an already-standardised frame into identifiers / features / labels."""
    rows_read = len(df)
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not feature_cols:
        raise ValueError(
            "No CICFlowMeter feature columns found - is this a CICFlowMeter / CICIDS2017 CSV?"
        )
    # Fully blank lines (the real Thursday file has ~288k of them).
    empty = df[feature_cols].isna().all(axis=1)
    df = df.loc[~empty].reset_index(drop=True)

    notes = []
    ids = pd.DataFrame(index=df.index)
    for col in ID_COLUMNS + ["Destination Port"]:
        if col in df.columns:
            ids[col] = df[col]
        else:
            ids[col] = np.nan
            if col != "Flow ID":
                notes.append(f"column '{col}' missing - logged as unknown")
    ids["Timestamp"] = ids["Timestamp"].astype(str).where(ids["Timestamp"].notna(), "")
    ids["captured_at"] = parse_timestamps(ids["Timestamp"]) if (ids["Timestamp"] != "").any() \
        else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    features = coerce_features(df, feature_cols, float_dtype)
    labels = categories = None
    if LABEL_COLUMN in df.columns:
        labels = df[LABEL_COLUMN].astype(str).map(canonical_label)
        categories = labels.map(label_to_category)
    return FlowBatch(ids=ids, features=features, labels=labels, categories=categories,
                     source_name=source_name, rows_read=rows_read,
                     rows_dropped_empty=int(empty.sum()), notes=notes)


def read_flow_csv(src, source_name: str | None = None, max_rows: int | None = None,
                  float_dtype=np.float64) -> FlowBatch:
    """Read one CICFlowMeter / CICIDS2017 CSV (path, bytes or file object)."""
    name = source_name or (Path(src).name if isinstance(src, (str, Path)) else "upload")
    df = read_csv_any_encoding(src, nrows=max_rows)
    return split_frame(standardise_columns(df), name, float_dtype)


def ip_str(value) -> str:
    return "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value).strip()


def int_or_none(value):
    try:
        v = float(value)
        return None if np.isnan(v) else int(v)
    except (TypeError, ValueError):
        return None


def utcnow() -> datetime:
    """Naive UTC 'now' - the database stores naive UTC datetimes throughout."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def to_datetime_or_now(value) -> datetime:
    if value is None or pd.isna(value):
        return utcnow()
    return pd.Timestamp(value).to_pydatetime()
