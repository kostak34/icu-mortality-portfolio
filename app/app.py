# app/app.py
from __future__ import annotations

import sys
import tempfile
import hashlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Path setup (robust imports)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step_01_load_raw import load_patient_long  # type: ignore
from scripts.step_02_batch_features import summarise_patient  # type: ignore

DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "outputs" / "model_bundle.joblib"
MAX_UPLOAD_BYTES = 2_000_000  # 2MB


# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def load_bundle(bundle_path: Path) -> dict[str, Any]:
    return joblib.load(bundle_path)


def short_sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def align_and_impute(
    feats: dict[str, Any], feature_columns: list[str], imputer
) -> tuple[pd.DataFrame, int, list[str], pd.DataFrame]:
    """Return (imputed_X, present_count, missing_features, aligned_pre_impute_X)."""
    X = pd.DataFrame([feats])

    for c in feature_columns:
        if c not in X.columns:
            X[c] = np.nan

    X = X[feature_columns]

    missing_mask = X.isna().iloc[0]
    missing_features = X.columns[missing_mask].tolist()
    present_count = int((~missing_mask).sum())

    X = X.where(pd.notna(X), np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")

    X_imp = pd.DataFrame(imputer.transform(X), columns=feature_columns)
    return X_imp, present_count, missing_features, X


def risk_band(prob: float, thr_medium: float, thr_high: float) -> str:
    if prob < thr_medium:
        return "LOW"
    if prob < thr_high:
        return "MEDIUM"
    return "HIGH"


def band_emoji(band: str) -> str:
    return {"LOW": "🟢", "MEDIUM": "🟠", "HIGH": "🔴"}[band]


def fmt_pct(prob: float) -> str:
    # Avoid ugly 0.000000 presentation
    if prob < 0.001:
        return "<0.10%"
    return f"{prob*100:.2f}%"


# -----------------------------
# Clinician-friendly field config (Quick entry)
# -----------------------------
# We map user-friendly fields to engineered feature names.
# For quick-entry simplicity:
# - we populate both *_last and *_mean from one entered value
# - we set *_was_measured = 1 when the value is provided
# - we map ventilation yes/no to MechVent_last and MechVent_prop_on
QUICK_FIELDS = [
    # key, label, unit, min, max, step, maps_to (last, mean, was_measured)
    ("RespRate", "Respiratory rate", "breaths/min", 0.0, 80.0, 1.0, ("RespRate_last", "RespRate_mean", "RespRate_was_measured")),
    ("SaO2", "Oxygen saturation (SpO₂)", "%", 0.0, 100.0, 1.0, ("SaO2_last", "SaO2_mean", "SaO2_was_measured")),
    ("FiO2", "FiO₂", "fraction (e.g., 0.21)", 0.21, 1.0, 0.01, ("FiO2_last", "FiO2_mean", "FiO2_was_measured")),
    ("NIMAP", "Mean arterial pressure (MAP)", "mmHg", 0.0, 200.0, 1.0, ("NIMAP_last", "NIMAP_mean", "NIMAP_was_measured")),
    ("NIDiasABP", "Diastolic BP", "mmHg", 0.0, 200.0, 1.0, ("NIDiasABP_last", "NIDiasABP_mean", "NIDiasABP_was_measured")),
    ("Lactate", "Lactate", "mmol/L", 0.0, 30.0, 0.1, ("Lactate_last", "Lactate_mean", "Lactate_was_measured")),
    ("pH", "Arterial pH", "", 6.8, 7.8, 0.01, ("pH_last", "pH_mean", "pH_was_measured")),
    ("Glucose", "Glucose", "mmol/L", 0.0, 60.0, 0.1, ("Glucose_last", "Glucose_mean", "Glucose_was_measured")),
]

# Optional: a small set of advanced engineered features you’ll commonly want
ADVANCED_SUGGESTED = [
    "HR_last", "HR_mean",
    "SysBP_last", "SysBP_mean",
    "Temp_last", "Temp_mean",
    "GCS_last", "GCS_mean",
    "MechVent_last", "MechVent_prop_on",
]


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="ICU Mortality Risk Demo", layout="centered")
st.title("ICU Mortality Risk Demo (Portfolio App)")
st.caption("Educational portfolio demo only — not for clinical use.")

with st.expander("What this is (and isn’t)"):
    st.write(
        """
This app demonstrates an end-to-end ML workflow (parse → features → impute → predict → risk bands).
It is **not** a validated clinical device, and the risk bands are **demo thresholds**.
        """
    )

bundle_path = DEFAULT_BUNDLE_PATH
if not bundle_path.exists():
    st.error(f"Model bundle not found: {bundle_path}. Make sure outputs/model_bundle.joblib is committed.")
    st.stop()

bundle = load_bundle(bundle_path)
model = bundle["model"]
imputer = bundle["imputer"]
feature_columns: list[str] = bundle["feature_columns"]

thr_high = float(bundle.get("default_threshold", 0.219))
thr_medium = float(bundle.get("medium_threshold", 0.100))

starter_vars = bundle.get("starter_vars", None)

# A clean legend (no truncation)
st.subheader("Risk bands")
st.markdown(
    f"{band_emoji('LOW')} **Low** < {thr_medium:.3f} &nbsp;&nbsp;·&nbsp;&nbsp; "
    f"{band_emoji('MEDIUM')} **Medium** {thr_medium:.3f}–{thr_high:.3f} &nbsp;&nbsp;·&nbsp;&nbsp; "
    f"{band_emoji('HIGH')} **High** ≥ {thr_high:.3f}"
)

st.divider()
st.subheader("Input")

tab_upload, tab_manual = st.tabs(["Upload patient file (.txt)", "Manual entry (clinician-friendly)"])

# Storage for the chosen feature dict + source label
feats: dict[str, Any] | None = None
source_label = ""

with tab_upload:
    st.caption("Upload a single patient `.txt` file. Max size for this demo: 2MB.")
    uploaded = st.file_uploader("Upload patient file (.txt)", type=["txt"])

    if uploaded is not None:
        if uploaded.size > MAX_UPLOAD_BYTES:
            st.error("File too large for this demo (max 2MB).")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = Path(tmp.name)

            st.write(f"**File received:** `{uploaded.name}`")
            try:
                long_df = load_patient_long(tmp_path)
                feats = summarise_patient(long_df)
                source_label = "Upload (.txt)"
            except Exception as e:
                st.error("Could not parse and featurise this file.")
                st.exception(e)
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

with tab_manual:
    st.caption(
        "Enter a few common values. Anything left blank will be imputed by the model (demo behaviour)."
    )

    with st.form("quick_entry_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        entered: dict[str, Any] = {}

        # Ventilation as a proper yes/no
        with col1:
            vent_on = st.selectbox("Invasive mechanical ventilation?", ["Unknown", "No", "Yes"], index=0)
        with col2:
            st.caption("")  # spacer

        # Numeric clinician fields
        for i, (key, label, unit, vmin, vmax, step, _) in enumerate(QUICK_FIELDS):
            col = col1 if i % 2 == 0 else col2
            with col:
                help_txt = f"Units: {unit}" if unit else None
                val = st.number_input(
                    f"{label}",
                    min_value=float(vmin),
                    max_value=float(vmax),
                    value=None,
                    step=float(step),
                    format="%.2f" if step < 1 else "%.0f",
                    help=help_txt,
                )
                if val is not None:
                    entered[key] = float(val)

        submitted = st.form_submit_button("Run prediction")

    if submitted:
        feats = {}

        # Map ventilation choice
        if vent_on == "Yes":
            feats["MechVent_last"] = 1.0
            feats["MechVent_prop_on"] = 1.0
            feats["MechVent_was_measured"] = 1.0
        elif vent_on == "No":
            feats["MechVent_last"] = 0.0
            feats["MechVent_prop_on"] = 0.0
            feats["MechVent_was_measured"] = 1.0
        else:
            # Unknown → leave missing (imputed)
            pass

        # Map quick numeric fields to engineered features
        for key, _, _, _, _, _, (last_name, mean_name, measured_name) in QUICK_FIELDS:
            if key in entered:
                v = entered[key]
                feats[last_name] = v
                feats[mean_name] = v
                feats[measured_name] = 1.0

        source_label = "Manual entry"

        # Optional: Advanced section (still manual, but portfolio-friendly)
        with st.expander("Advanced (portfolio): add more engineered features"):
            suggested = [c for c in ADVANCED_SUGGESTED if c in feature_columns]
            extra_cols = st.multiselect(
                "Choose extra engineered features to enter",
                options=feature_columns,
                default=suggested,
            )
            for c in extra_cols:
                if c in feats:
                    continue
                # Use numeric input for anything that isn't obviously boolean-ish
                feats[c] = st.number_input(c, value=None)

# Only proceed if we have features from either tab
if feats is None:
    st.stop()

# Predict
try:
    X_imp, present_count, missing_features, X_aligned = align_and_impute(feats, feature_columns, imputer)
    prob = float(model.predict_proba(X_imp)[:, 1][0])
    band = risk_band(prob, thr_medium, thr_high)

    st.divider()
    st.subheader("Prediction")
    st.caption(f"Source: **{source_label}**")

    st.metric("Predicted mortality risk", fmt_pct(prob))
    st.progress(min(max(prob, 0.0), 1.0))

    st.write(f"**Risk band:** {band_emoji(band)} **{band} RISK**")
    st.write(f"**High-risk alert:** {'YES' if band == 'HIGH' else 'NO'}")
    st.caption(f"Raw probability: {prob:.6f}")

    st.subheader("Data completeness")
    st.write(f"**Features present:** {present_count}/{len(feature_columns)}")
    st.write(f"**Missing (imputed):** {len(missing_features)}")

    with st.expander("Show missing features"):
        # make it readable, not a JSON-looking blob
        for f in missing_features:
            st.markdown(f"- `{f}`")

    st.subheader("Engineered features")
    show_all = st.checkbox("Show ALL engineered features (including missing/null)", value=False)

    # Show aligned features pre-imputation, but only non-null by default
    X_row = X_aligned.iloc[0].copy()
    if not show_all:
        X_row = X_row.dropna()

    engineered_df = pd.DataFrame({"feature": X_row.index, "value": X_row.values})
    st.dataframe(engineered_df, use_container_width=True, hide_index=True)

    with st.expander("Technical details (for portfolio / debugging)"):
        st.write(f"- Bundle path (repo): `{DEFAULT_BUNDLE_PATH.as_posix()}`")
        if DEFAULT_BUNDLE_PATH.exists():
            st.write(f"- Bundle SHA256 (short): `{short_sha256_of_file(DEFAULT_BUNDLE_PATH)}`")
        st.write(f"- Thresholds: LOW < {thr_medium:.3f}, MEDIUM [{thr_medium:.3f}..{thr_high:.3f}), HIGH ≥ {thr_high:.3f}")
        if starter_vars is not None:
            st.write(f"- starter_vars: {len(starter_vars)} variables")
        st.write(f"- Python: {sys.version.split()[0]}")
        st.write(f"- numpy: {np.__version__}")
        st.write(f"- pandas: {pd.__version__}")
        st.write(f"- joblib: {joblib.__version__}")

except Exception as e:
    st.error("Something went wrong while predicting.")
    st.exception(e)
