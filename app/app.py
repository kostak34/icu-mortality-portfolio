"""
Streamlit demo app (educational portfolio).

Run locally (after installing deps):
  python -m streamlit run app/app.py

Notes:
- Requires a saved model bundle at outputs/model_bundle.joblib
- Supports two input modes:
  1) Upload a single patient .txt file (parsed -> engineered features)
  2) Manual entry (optional): enter a small set of observations; the app derives mean/last
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from scripts.step_01_load_raw import load_patient_long  # type: ignore
from scripts.step_02_batch_features import summarise_patient  # type: ignore


# -----------------------------
# Page config (must be first)
# -----------------------------
st.set_page_config(page_title="ICU Mortality Risk Demo", layout="centered")


# -----------------------------
# Helpers
# -----------------------------
BUNDLE_PATH = Path("outputs/model_bundle.joblib")


def short_sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:10]


def safe_float(s: str) -> Optional[float]:
    """
    Parse a user-entered string into float.
    - Blank -> None
    - Commas allowed (e.g., "7,2" -> 7.2)
    """
    if s is None:
        return None
    t = str(s).strip()
    if t == "":
        return None
    t = t.replace(",", ".")
    return float(t)


def align_and_impute(
    feats: Dict[str, Any],
    feature_columns: List[str],
    imputer: Any,
) -> Tuple[pd.DataFrame, int, List[str], pd.DataFrame]:
    """
    Align to training feature columns, count present features, list missing features, and impute.
    Returns:
      X_imp, present_count, missing_features, X_aligned
    """
    X = pd.DataFrame([feats])

    # Align columns (missing -> NA)
    for c in feature_columns:
        if c not in X.columns:
            X[c] = pd.NA
    X = X[feature_columns]

    present_mask = ~X.isna().iloc[0]
    present_count = int(present_mask.sum())
    missing_features = [c for c in feature_columns if not bool(present_mask[c])]

    # Impute (imputer expects numeric-ish)
    X_imp = pd.DataFrame(imputer.transform(X), columns=feature_columns)
    return X_imp, present_count, missing_features, X


def risk_band(prob: float, thr_med: float, thr_high: float) -> str:
    if prob >= thr_high:
        return "HIGH"
    if prob >= thr_med:
        return "MEDIUM"
    return "LOW"


def band_badge(band: str) -> str:
    # Simple emoji badge to avoid heavy styling
    return {"LOW": "🟢", "MEDIUM": "🟠", "HIGH": "🔴"}.get(band, "⚪")


def group_feature_bases(feature_columns: List[str]) -> List[str]:
    """
    Group engineered features into base variable names by removing common suffixes.
    This lets manual entry present clinician-friendly 'variables' rather than raw column names.
    """
    suffixes = (
        "_last",
        "_mean",
        "_min",
        "_max",
        "_std",
        "_was_measured",
        "_prop_on",
    )
    bases = set()
    for c in feature_columns:
        base = c
        for sfx in suffixes:
            if base.endswith(sfx):
                base = base[: -len(sfx)]
                break
        bases.add(base)
    # Drop empty base (just in case)
    bases.discard("")
    return sorted(bases)


CLINICAL_META: Dict[str, Dict[str, str]] = {
    "RespRate": {"label": "Respiratory rate", "unit": "breaths/min"},
    "HR": {"label": "Heart rate", "unit": "bpm"},
    "SaO2": {"label": "Oxygen saturation (SpO₂)", "unit": "%"},
    "FiO2": {"label": "FiO₂", "unit": "fraction (e.g., 0.21)"},
    "Lactate": {"label": "Lactate", "unit": "mmol/L"},
    "pH": {"label": "Arterial pH", "unit": ""},
    "Glucose": {"label": "Glucose", "unit": "mmol/L"},
    "NIMAP": {"label": "Non-invasive MAP", "unit": "mmHg"},
    "NIDiasABP": {"label": "Non-invasive diastolic BP", "unit": "mmHg"},
    "NISysABP": {"label": "Non-invasive systolic BP", "unit": "mmHg"},
    "MechVent": {"label": "Invasive ventilation", "unit": ""},
}


def pretty_base_name(base: str) -> str:
    meta = CLINICAL_META.get(base)
    if meta:
        unit = meta.get("unit", "")
        return f"{meta.get('label', base)}" + (f" ({unit})" if unit else "")
    # fall back to raw base if unknown
    return base


def manual_entry_build_features(
    chosen_bases: List[str],
    feature_columns: List[str],
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Manual entry UI:
    - For numeric variables: collect up to 3 readings (earlier -> now), compute mean + last
    - For boolean-ish variables (MechVent): collect earlier/now as Yes/No/Unknown and compute
    Returns feats dict (engineered features) + list of user-facing warnings.
    """
    feats: Dict[str, Any] = {}
    warnings: List[str] = []

    st.caption(
        "Enter a few observations. You can leave anything blank if unknown. "
        "This demo will still run, but missing values may reduce reliability."
    )

    for base in chosen_bases:
        # Determine which engineered columns exist for this base
        has_last = f"{base}_last" in feature_columns
        has_mean = f"{base}_mean" in feature_columns
        has_was = f"{base}_was_measured" in feature_columns
        has_prop = f"{base}_prop_on" in feature_columns

        # If none exist, skip (unlikely but safe)
        if not any([has_last, has_mean, has_was, has_prop]):
            continue

        st.markdown(f"### {pretty_base_name(base)}")

        if base == "MechVent":
            c1, c2 = st.columns(2)
            earlier = c1.selectbox(
                "Earlier status",
                ["Unknown", "No", "Yes"],
                index=0,
                key=f"{base}_earlier",
                help="If you know whether the patient was invasively ventilated earlier in the window.",
            )
            now = c2.selectbox(
                "Most recent status",
                ["Unknown", "No", "Yes"],
                index=0,
                key=f"{base}_now",
                help="Current/most recent status.",
            )

            def map_bool(x: str) -> Optional[float]:
                if x == "Yes":
                    return 1.0
                if x == "No":
                    return 0.0
                return None

            v_earlier = map_bool(earlier)
            v_now = map_bool(now)

            vals = [v for v in [v_earlier, v_now] if v is not None]

            if has_last:
                # Prefer "now", else fall back to earlier
                last_val = v_now if v_now is not None else v_earlier
                feats[f"{base}_last"] = last_val if last_val is not None else pd.NA
            if has_mean:
                feats[f"{base}_mean"] = float(np.mean(vals)) if vals else pd.NA
            if has_prop:
                # In this simplified manual mode, treat proportion-on as mean of known statuses.
                feats[f"{base}_prop_on"] = float(np.mean(vals)) if vals else pd.NA
            if has_was:
                feats[f"{base}_was_measured"] = 1 if vals else 0

            continue  # done with MechVent

        # Numeric-ish variables
        unit_hint = CLINICAL_META.get(base, {}).get("unit", "")
        hint = f"Example: 7.2" + (f" ({unit_hint})" if unit_hint else "")

        col1, col2, col3 = st.columns(3)
        s1 = col1.text_input(
            "Earlier (e.g., 12h ago)",
            value="",
            placeholder=hint,
            key=f"{base}_t12",
        )
        s2 = col2.text_input(
            "Earlier (e.g., 6h ago)",
            value="",
            placeholder=hint,
            key=f"{base}_t6",
        )
        s3 = col3.text_input(
            "Most recent",
            value="",
            placeholder=hint,
            key=f"{base}_t0",
        )

        parsed: List[Optional[float]] = []
        for label, s in [("earlier (12h)", s1), ("earlier (6h)", s2), ("most recent", s3)]:
            try:
                parsed.append(safe_float(s))
            except Exception:
                warnings.append(
                    f"'{pretty_base_name(base)}' – please enter numbers only (or leave blank). "
                    f"The {label} value was ignored."
                )
                parsed.append(None)

        # Values in chronological order; determine last by most-recent-first fallback
        v12, v6, v0 = parsed
        vals = [v for v in [v12, v6, v0] if v is not None]

        if has_last:
            last_val = v0 if v0 is not None else (v6 if v6 is not None else v12)
            feats[f"{base}_last"] = last_val if last_val is not None else pd.NA

        if has_mean:
            feats[f"{base}_mean"] = float(np.mean(vals)) if vals else pd.NA

        if has_was:
            feats[f"{base}_was_measured"] = 1 if vals else 0

        # If the model expects prop_on for a numeric base (rare), ignore it here.

    return feats, warnings


# -----------------------------
# Load bundle
# -----------------------------
if not BUNDLE_PATH.exists():
    st.error("Model bundle not found at outputs/model_bundle.joblib.")
    st.stop()

bundle = joblib.load(BUNDLE_PATH)
model = bundle["model"]
imputer = bundle["imputer"]
feature_columns: List[str] = list(bundle["feature_columns"])

thr_high = float(bundle.get("high_threshold", bundle.get("default_threshold", 0.219)))
thr_med = float(bundle.get("medium_threshold", 0.100))

# Starter vars are nice for portfolio context; keep available but don’t push into clinician UI
starter_vars = bundle.get("starter_vars", None)
bundle_hash_short = short_sha256_of_file(BUNDLE_PATH)


# -----------------------------
# UI: Header + expander copy (Option A)
# -----------------------------
st.title("ICU Mortality Risk Demo (Portfolio App)")
st.caption("Educational portfolio demo only — not for clinical use.")

with st.expander("What this tool does (and doesn’t)"):
    st.markdown(
        """
**What this tool does**  
Upload (or enter) a small set of observations and the app returns an estimated risk band (Low / Medium / High).

**How to interpret it**  
Treat the output as a **demonstration only**. It is not a validated clinical decision tool and must not be used to guide patient care.

**Why you might still look at it**  
It shows what an ML-assisted risk score *could* look like in a future workflow, including how missing data is handled.
"""
    )

# A small “model status” row that is readable (no truncation)
st.markdown("## Model status")
c1, c2, c3 = st.columns(3)
c1.markdown("**Risk bands**  \nLow / Medium / High")
c2.metric("Features expected", f"{len(feature_columns)}")
c3.metric("Model ID", bundle_hash_short)


# -----------------------------
# Input modes
# -----------------------------
st.markdown("## Input")

tab_upload, tab_manual = st.tabs(["Upload patient file (.txt)", "Manual entry (optional)"])

source_label = None
raw_engineered: Optional[Dict[str, Any]] = None

with tab_upload:
    st.caption("Upload a single patient .txt file. (This demo recommends keeping files small.)")

    uploaded = st.file_uploader("Upload patient file (.txt)", type=["txt"])
    if uploaded is not None:
        # Soft limit messaging (Streamlit Cloud may show a bigger global limit)
        size_bytes = len(uploaded.getvalue())
        if size_bytes > 2 * 1024 * 1024:
            st.warning("This file is larger than 2MB. For this demo, consider using smaller example files.")

        tmp_path = Path("outputs/_tmp_uploaded_patient.txt")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(uploaded.getvalue())

        try:
            long_df = load_patient_long(tmp_path)
            feats = summarise_patient(long_df)
            raw_engineered = feats
            source_label = "File upload"
            st.success(f"File received: {uploaded.name}")
        except Exception as e:
            st.error("Could not process that file.")
            st.exception(e)

with tab_manual:
    st.caption(
        "Manual entry is a convenience mode for the demo. "
        "You can enter a few observations; the model will fill the rest."
    )

    all_bases = group_feature_bases(feature_columns)

    # Prefer a sensible default set if present
    preferred = [b for b in ["RespRate", "SaO2", "FiO2", "Lactate", "pH", "HR", "Glucose", "NIMAP", "NIDiasABP", "MechVent"] if b in all_bases]
    default_selected = preferred[:8] if preferred else all_bases[:8]

    chosen_bases = st.multiselect(
        "Choose which observations to enter",
        options=all_bases,
        default=default_selected,
        format_func=pretty_base_name,
        help="Choose a small set. Leaving values blank is allowed.",
    )

    feats_manual, warnings = manual_entry_build_features(chosen_bases, feature_columns)

    if warnings:
        for w in warnings:
            st.warning(w)

    run_manual = st.button("Run prediction", type="primary")
    if run_manual:
        raw_engineered = feats_manual
        source_label = "Manual entry"


# -----------------------------
# Predict + display
# -----------------------------
if raw_engineered is None or source_label is None:
    st.stop()

try:
    X_imp, present_count, missing_features, X_aligned = align_and_impute(raw_engineered, feature_columns, imputer)
    prob = float(model.predict_proba(X_imp)[:, 1][0])
    band = risk_band(prob, thr_med=thr_med, thr_high=thr_high)
    high_alert = prob >= thr_high

    st.markdown("## Prediction")
    st.caption(f"Source: {source_label}")

    st.markdown("**Predicted mortality risk**")
    st.markdown(f"<span style='font-size:42px; font-weight:700;'>{prob*100:.2f}%</span>", unsafe_allow_html=True)
    st.progress(min(max(prob, 0.0), 1.0))

    st.markdown(f"**Risk band:** {band_badge(band)} **{band} RISK**")
    st.markdown(f"**High-risk alert:** {'YES' if high_alert else 'NO'}")
    st.caption(f"Raw probability: {prob:.6f}")

    st.markdown("## Data completeness")
    st.write(f"Features present: **{present_count}/{len(feature_columns)}**")
    st.write(f"Missing features (filled by the demo): **{len(missing_features)}**")

    with st.expander("Show missing features"):
        # Clinician-friendly: list, not JSON-ish blob
        st.write(missing_features)

    # Engineered features: show non-missing by default, optionally show all
    st.markdown("## Engineered features")

    show_all = st.checkbox("Show ALL engineered features (including missing/blank)", value=False)

    # Convert one-row aligned DF to long
    long = X_aligned.iloc[0].reset_index()
    long.columns = ["feature", "value"]

    if not show_all:
        long = long[~pd.isna(long["value"])]

    with st.expander("Show engineered features (aligned, pre-imputation)"):
        st.dataframe(long, use_container_width=True, hide_index=True)

    # A small, plain-English note for manual mode
    if source_label == "Manual entry":
        st.warning(
            "Manual entry mode used: some values were left blank and the demo filled the gaps. "
            "That is expected in this prototype, but it may affect the result."
        )

    # Keep technical details available for portfolio/debugging, but out of the main flow
    with st.expander("Technical details (for portfolio / debugging)"):
        st.write(f"- Bundle path (repo): {str(BUNDLE_PATH)}")
        st.write(f"- Bundle SHA256 (short): {bundle_hash_short}")
        st.write(f"- Thresholds: LOW < {thr_med:.3f}, MEDIUM [{thr_med:.3f}..{thr_high:.3f}), HIGH ≥ {thr_high:.3f}")
        if starter_vars is not None:
            try:
                st.write(f"- starter_vars: {len(starter_vars)} variables")
            except Exception:
                st.write("- starter_vars: present")
        st.write(f"- Python: {sys.version.split()[0]}")
        try:
            import sklearn  # noqa
            st.write(f"- scikit-learn: {sklearn.__version__}")  # type: ignore
        except Exception:
            pass
        st.write(f"- numpy: {np.__version__}")
        st.write(f"- pandas: {pd.__version__}")
        st.write(f"- joblib: {joblib.__version__}")

except Exception as e:
    st.error("Something went wrong while predicting.")
    st.exception(e)
