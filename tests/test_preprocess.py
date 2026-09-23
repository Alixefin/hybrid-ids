"""Pre-processing pipeline edge cases."""
import numpy as np
import pandas as pd
import pytest

from engine.ingest import parse_timestamps, read_flow_csv, standardise_columns
from engine.preprocess import FlowPreprocessor, clean_rows, signed_log1p
from engine.schema import FEATURE_COLUMNS, canonical_label, label_to_category


def _frame(n=400, seed=0):
    """Minimal synthetic feature frame with controlled properties."""
    rng = np.random.default_rng(seed)
    y = np.where(rng.random(n) < 0.8, "Benign", "DoS/DDoS")
    attack = y != "Benign"
    dur = np.where(attack, rng.lognormal(12, 0.3, n), rng.lognormal(10, 1.5, n))
    fwd = rng.integers(1, 50, n).astype(float)
    df = pd.DataFrame({
        "Flow Duration": dur,
        "Total Fwd Packets": fwd,
        "Subflow Fwd Packets": fwd,                       # perfectly correlated duplicate
        "Fwd Packet Length Mean": np.where(attack, 900.0, 80.0) + rng.normal(0, 20, n),
        "Init_Win_bytes_forward": rng.choice([-1, 8192, 29200], n).astype(float),
        "Bwd PSH Flags": np.zeros(n),                     # constant
        "Flow Bytes/s": rng.lognormal(8, 2, n),
        "Destination Port": rng.choice([80, 443, 53], n).astype(float),
    })
    return df, pd.Series(y)


def test_clean_rows_drops_nan_inf_and_duplicates():
    df, y = _frame(50)
    df["Label"] = y
    df.loc[0, "Flow Bytes/s"] = np.inf
    df.loc[1, "Flow Duration"] = np.nan
    df.loc[2, "Total Fwd Packets"] = -np.inf
    df = pd.concat([df, df.iloc[[10, 11]]], ignore_index=True)  # two exact duplicates
    out, stats = clean_rows(df, "Label")
    assert stats["dropped_non_finite"] == 3
    assert stats["dropped_duplicates"] == 2
    assert len(out) == 50 - 3
    assert np.isfinite(out[[c for c in FEATURE_COLUMNS if c in out]].to_numpy()).all()


def test_clean_rows_parses_infinity_strings():
    df, y = _frame(10)
    df = df.astype(object)
    df.loc[3, "Flow Bytes/s"] = "Infinity"
    df.loc[4, "Flow Bytes/s"] = "NaN"
    df.loc[5, "Flow Bytes/s"] = "not a number"
    out, stats = clean_rows(df, None)
    assert stats["dropped_non_finite"] == 3 and len(out) == 7


def test_clean_rows_ignores_all_missing_column():
    df, y = _frame(20)
    df["Idle Mean"] = np.nan  # column with no value at all must not wipe every row
    out, stats = clean_rows(df, None)
    assert len(out) == 20
    assert stats["ignored_all_missing_columns"] == ["Idle Mean"]


def test_clean_rows_without_feature_columns_raises():
    with pytest.raises(ValueError):
        clean_rows(pd.DataFrame({"foo": [1, 2]}), None)


def test_fit_drops_constant_and_correlated_and_selects_top_k():
    df, y = _frame()
    pre = FlowPreprocessor(n_features=3, rf_estimators=20).fit(df, y)
    assert "Bwd PSH Flags" in pre.dropped_constant_
    assert pre.dropped_correlated_.get("Subflow Fwd Packets") == "Total Fwd Packets"
    assert len(pre.selected_features_) == 3
    assert "Bwd PSH Flags" not in pre.selected_features_
    # selected features keep CICFlowMeter column order
    order = [FEATURE_COLUMNS.index(c) for c in pre.selected_features_]
    assert order == sorted(order)
    # the discriminative features are found
    assert {"Fwd Packet Length Mean", "Flow Duration"} <= set(pre.selected_features_)


def test_all_missing_feature_is_treated_as_constant():
    df, y = _frame()
    df["Idle Mean"] = np.nan
    pre = FlowPreprocessor(n_features=10, rf_estimators=10).fit(df, y)
    assert "Idle Mean" in pre.dropped_constant_


def test_all_constant_input_raises():
    df = pd.DataFrame({"Flow Duration": np.ones(20), "Total Fwd Packets": np.zeros(20)})
    with pytest.raises(ValueError):
        FlowPreprocessor(rf_estimators=5).fit(df, ["Benign"] * 20)


def test_transform_clips_logs_and_repairs_non_finite():
    df, y = _frame()
    pre = FlowPreprocessor(n_features=10, rf_estimators=10).fit(df, y)
    cols = pre.selected_features_
    test = df.iloc[:4].copy()
    test.loc[test.index[0], "Flow Bytes/s"] = np.inf
    test.loc[test.index[1], "Flow Bytes/s"] = np.nan
    test.loc[test.index[2], "Flow Duration"] = 1e30          # beyond training range
    X, repaired = pre.transform(test, return_repaired=True)
    assert X.shape == (4, len(cols)) and X.dtype == np.float32
    assert np.isfinite(X).all()
    assert repaired.tolist() == [True, True, False, False]
    j = cols.index("Flow Duration")
    assert X[2, j] == pytest.approx(signed_log1p(np.array([pre.clip_upper_["Flow Duration"]]))[0], rel=1e-5)
    if "Flow Bytes/s" in cols:
        k = cols.index("Flow Bytes/s")
        assert X[0, k] == pytest.approx(signed_log1p(np.array([pre.clip_upper_["Flow Bytes/s"]]))[0], rel=1e-5)


def test_signed_log_handles_negative_window_sizes():
    out = signed_log1p(np.array([-1.0, 0.0, 29200.0]))
    assert out[0] == pytest.approx(-np.log(2)) and out[1] == 0.0 and out[2] > 10


def test_transform_missing_column_raises_clear_error():
    df, y = _frame()
    pre = FlowPreprocessor(n_features=5, rf_estimators=10).fit(df, y)
    with pytest.raises(ValueError, match="missing"):
        pre.transform(df.drop(columns=[pre.selected_features_[0]]))


def test_transform_ignores_extra_columns_and_empty_frame():
    df, y = _frame()
    pre = FlowPreprocessor(n_features=5, rf_estimators=10).fit(df, y)
    extra = df.assign(**{"Source IP": "1.2.3.4", "Unrelated": 7})
    assert pre.transform(extra).shape == (len(df), 5)
    assert pre.transform(df.iloc[:0]).shape == (0, 5)


def test_unfitted_transform_raises():
    with pytest.raises(RuntimeError):
        FlowPreprocessor().transform(pd.DataFrame({"Flow Duration": [1.0]}))


# --- ingestion-level normalisation ------------------------------------------------------
def test_raw_cicids_headers_are_normalised():
    raw = pd.DataFrame([[1, 2, 3]], columns=[" Destination Port", " Fwd Header Length", "Flow Bytes/s"])
    out = standardise_columns(raw)
    assert list(out.columns)[:3] == ["Destination Port", "Fwd Header Length", "Flow Bytes/s"]
    assert "Fwd Header Length.1" in out  # restored duplicate column


def test_cicflowmeter_v4_aliases(tmp_path):
    p = tmp_path / "v4.csv"
    pd.DataFrame({"Src IP": ["10.0.0.1"], "Dst IP": ["10.0.0.2"], "Dst Port": [80], "Flow Duration": [10],
                  "Tot Fwd Pkts": [3], "Flow Byts/s": ["Infinity"]}).to_csv(p, index=False)
    batch = read_flow_csv(p)
    assert batch.ids.loc[0, "Source IP"] == "10.0.0.1"
    assert batch.features.loc[0, "Total Fwd Packets"] == 3
    assert np.isposinf(batch.features.loc[0, "Flow Bytes/s"])
    assert not batch.has_labels


def test_latin1_file_with_cp1252_dash(tmp_path):
    p = tmp_path / "thursday.csv"
    p.write_bytes(" Flow Duration, Label\n5,Web Attack \x96 XSS\n7,BENIGN\n,\n".encode("latin-1"))
    batch = read_flow_csv(p)
    assert batch.labels.tolist() == ["Web Attack - XSS", "BENIGN"]
    assert batch.rows_dropped_empty == 1
    assert batch.categories.tolist() == ["Web Attack", "Benign"]


def test_non_flow_csv_rejected(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="CICFlowMeter"):
        read_flow_csv(p)


def test_twelve_hour_timestamps_are_repaired():
    ts = parse_timestamps(pd.Series(["7/7/2017 3:56", "7/7/2017 9:05:01", "7/7/2017 3:56 PM", "2017-07-07 15:56:00"]))
    assert [t.hour for t in ts] == [15, 9, 15, 15]
    assert ts.iloc[0].day == 7 and ts.iloc[0].month == 7


@pytest.mark.parametrize("raw,canon,cat", [
    ("BENIGN", "BENIGN", "Benign"),
    ("Web Attack � Brute Force", "Web Attack - Brute Force", "Web Attack"),
    ("Web Attack – Sql Injection", "Web Attack - Sql Injection", "Web Attack"),
    ("DoS slowloris", "DoS slowloris", "DoS/DDoS"),
    ("  PortScan ", "PortScan", "Port Scan"),
    ("Bot", "Bot", "Botnet (Malware)"),
    ("Heartbleed", "Heartbleed", "Other (Infiltration/Heartbleed)"),
])
def test_label_mapping(raw, canon, cat):
    assert canonical_label(raw) == canon
    assert label_to_category(raw) == cat
