"""Pre-processing pipeline (fitted once at training time, saved inside the model file).

Training-time row cleaning  (``clean_rows``)
    * drop rows with missing / infinite feature values
    * drop duplicate flows (identical features + label), *before* any train/test split
      so duplicates cannot leak across splits

Column pipeline  (``FlowPreprocessor.fit`` / ``transform``)
    1. identifier / time columns never enter the model (they are kept in the FlowBatch
       for logging only)
    2. drop near-constant features (one value covers >= 99.95 % of rows, or zero
       variance after clipping)
    3. clip every feature to the [min, 99.9th percentile] range seen in training
    4. signed log1p on heavy-tailed count / byte / time features
    5. drop highly correlated features (|Pearson r| > 0.95 after transformation; the
       first column in CICFlowMeter order is kept)
    6. keep the top-k (default 30) features by Random Forest impurity importance

Inference-time repair
    At inference a flow cannot simply be dropped (a zero-duration port-scan probe
    yields Flow Bytes/s = Infinity), so non-finite values are *repaired* instead:
    +inf -> training upper clip bound, -inf -> lower bound, NaN -> 0 (then clipped).
    ``transform(..., return_repaired=True)`` reports which rows were repaired.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from engine.schema import FEATURE_COLUMNS, is_heavy_tailed


def _numeric(df: pd.DataFrame, columns) -> pd.DataFrame:
    return df[list(columns)].apply(pd.to_numeric, errors="coerce").astype(np.float64)


def clean_rows(df: pd.DataFrame, label_col: str | None = "Label",
               feature_cols=None) -> tuple[pd.DataFrame, dict]:
    """Drop rows with NaN / +-inf features and duplicate flows. Returns (frame, stats)."""
    cols = [c for c in (feature_cols or FEATURE_COLUMNS) if c in df.columns]
    if not cols:
        raise ValueError("no feature columns to clean")
    values = _numeric(df, cols)
    # A column with no finite value at all would otherwise remove every row: ignore it
    # here (the column pipeline drops it as constant).
    usable = [c for c in cols if np.isfinite(values[c].to_numpy()).any()]
    empty_cols = [c for c in cols if c not in usable]
    finite = np.isfinite(values[usable].to_numpy()).all(axis=1)
    out = df.loc[finite].copy()
    out[cols] = values.loc[finite]
    subset = usable + ([label_col] if label_col and label_col in out.columns else [])
    dup = out.duplicated(subset=subset, keep="first")
    out = out.loc[~dup]
    stats = {
        "rows_in": int(len(df)),
        "dropped_non_finite": int((~finite).sum()),
        "dropped_duplicates": int(dup.sum()),
        "rows_out": int(len(out)),
        "ignored_all_missing_columns": empty_cols,
    }
    return out, stats


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


class FlowPreprocessor:
    def __init__(self, n_features: int = 30, near_constant_share: float = 0.9995,
                 corr_threshold: float = 0.95, clip_quantile: float = 0.999,
                 rf_estimators: int = 100, max_fit_rows: int = 300_000,
                 random_state: int = 42):
        self.n_features = n_features
        self.near_constant_share = near_constant_share
        self.corr_threshold = corr_threshold
        self.clip_quantile = clip_quantile
        self.rf_estimators = rf_estimators
        self.max_fit_rows = max_fit_rows
        self.random_state = random_state

    # ------------------------------------------------------------------ fitting
    def fit(self, X: pd.DataFrame, y) -> "FlowPreprocessor":
        y = pd.Series(np.asarray(y), index=X.index)
        candidates = [c for c in FEATURE_COLUMNS if c in X.columns]
        if not candidates:
            raise ValueError("no CICFlowMeter feature columns in training data")
        raw = _numeric(X, candidates)
        all_missing = [c for c in candidates if not np.isfinite(raw[c].to_numpy()).any()]
        candidates = [c for c in candidates if c not in all_missing]
        raw = raw[candidates]
        finite = np.isfinite(raw.to_numpy()).all(axis=1)
        if not finite.all():
            raw, y = raw.loc[finite], y.loc[finite]
        if len(raw) < 2:
            raise ValueError("need at least 2 finite rows to fit the pre-processor")

        # 2. near-constant (on raw values); all-missing columns count as constant
        self.dropped_constant_ = list(all_missing)
        keep = []
        for c in candidates:
            top_share = raw[c].value_counts(normalize=True, dropna=False).iloc[0]
            if top_share >= self.near_constant_share:
                self.dropped_constant_.append(c)
            else:
                keep.append(c)

        # 3. clip bounds
        self.clip_lower_ = raw[keep].min().to_dict()
        self.clip_upper_ = raw[keep].quantile(self.clip_quantile).to_dict()
        # 4. log1p columns
        self.log_columns_ = [c for c in keep if is_heavy_tailed(c)]

        sample_idx = self._sample_index(y)
        T = self._transform_frame(raw.loc[sample_idx, keep], keep)

        # zero variance after clipping -> constant as well
        std = T.std(ddof=0)
        flat = std.index[std.to_numpy() == 0].tolist()
        self.dropped_constant_ += flat
        keep = [c for c in keep if c not in flat]
        if not keep:
            raise ValueError("every feature is constant or missing - nothing to learn from")

        # 5. correlation filter
        corr = T[keep].corr().abs().to_numpy()
        kept_idx, self.dropped_correlated_ = [], {}
        for j, c in enumerate(keep):
            partner = next((keep[i] for i in kept_idx if corr[i, j] > self.corr_threshold), None)
            if partner is None:
                kept_idx.append(j)
            else:
                self.dropped_correlated_[c] = partner
        keep = [keep[i] for i in kept_idx]

        # 6. Random Forest importance ranking
        rf = RandomForestClassifier(
            n_estimators=self.rf_estimators, class_weight="balanced", n_jobs=-1,
            max_depth=20, random_state=self.random_state,
        )
        rf.fit(T[keep].to_numpy(np.float32), y.loc[sample_idx].to_numpy())
        self.importances_ = (pd.Series(rf.feature_importances_, index=keep)
                             .sort_values(ascending=False))
        k = min(self.n_features, len(keep))
        top = set(self.importances_.index[:k])
        self.selected_features_ = [c for c in keep if c in top]  # CICFlowMeter order
        self.n_features_in_ = len(candidates) + len(all_missing)
        return self

    def _sample_index(self, y: pd.Series) -> pd.Index:
        if len(y) <= self.max_fit_rows:
            return y.index
        # stratified cap: keep every row of small classes, sample the large ones
        rng = np.random.default_rng(self.random_state)
        frac = self.max_fit_rows / len(y)
        parts = []
        for _, idx in y.groupby(y).groups.items():
            n = max(min(len(idx), 1000), int(round(len(idx) * frac)))
            parts.append(rng.choice(np.asarray(idx), size=min(n, len(idx)), replace=False))
        return pd.Index(np.concatenate(parts))

    # --------------------------------------------------------------- transform
    def _transform_frame(self, raw: pd.DataFrame, cols) -> pd.DataFrame:
        lo = pd.Series({c: self.clip_lower_[c] for c in cols})
        hi = pd.Series({c: self.clip_upper_[c] for c in cols})
        out = raw[cols].clip(lower=lo, upper=hi, axis=1)
        logs = [c for c in cols if c in self.log_columns_]
        if logs:
            out[logs] = signed_log1p(out[logs].to_numpy())
        return out

    def transform(self, X: pd.DataFrame, return_repaired: bool = False):
        self._check_fitted()
        cols = self.selected_features_
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"input is missing {len(missing)} required feature column(s): "
                             + ", ".join(missing))
        raw = _numeric(X, cols)
        arr = raw.to_numpy()
        repaired = ~np.isfinite(arr).all(axis=1)
        if repaired.any():
            hi = np.array([self.clip_upper_[c] for c in cols])
            lo = np.array([self.clip_lower_[c] for c in cols])
            arr = np.where(np.isposinf(arr), hi, arr)
            arr = np.where(np.isneginf(arr), lo, arr)
            arr = np.where(np.isnan(arr), 0.0, arr)
            raw = pd.DataFrame(arr, columns=cols, index=raw.index)
        out = self._transform_frame(raw, cols).to_numpy(np.float32)
        return (out, repaired) if return_repaired else out

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def _check_fitted(self):
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("FlowPreprocessor is not fitted")

    def report(self) -> dict:
        """Summary of what the pipeline kept / dropped (stored in model metadata)."""
        self._check_fitted()
        return {
            "n_input_features": self.n_features_in_,
            "dropped_near_constant": list(self.dropped_constant_),
            "dropped_correlated": dict(self.dropped_correlated_),
            "log1p_columns": [c for c in self.selected_features_ if c in self.log_columns_],
            "selected_features": list(self.selected_features_),
            "importances": {k: round(float(v), 5) for k, v in self.importances_.items()},
            "params": {
                "n_features": self.n_features, "near_constant_share": self.near_constant_share,
                "corr_threshold": self.corr_threshold, "clip_quantile": self.clip_quantile,
            },
        }
