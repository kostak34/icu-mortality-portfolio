# app/app.py
"""
ICU Mortality Risk Demo (Portfolio Streamlit App)

Clinician-facing demo:
- Upload mode: upload a PhysioNet-style patient .txt file → parse → engineer features → predict.
- Manual mode: enter a small set of clinical measurements → build approximate engineered features → predict.

Important:
- Educational / portfolio demo only. Not for clinical use.
- This app intentionally avoids reading any local/private data paths.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Page config (must be first Streamlit call)
# -----------------------------
st.set_page_config(page_title="ICU Mortality Risk Demo", layout="centered")


# -----------------------------
# Robust imports (works locally + on Streamlit Cloud)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repo root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.step_01_load_raw import load_patient_long  # type: ignore
    from scripts.step_02_batch_features import summarise_patient  # type: ignore
except Exception as e:
    st.error(
        "The app can't import its feature-engineering scripts. "
        "This usually means the `scripts/` package imports need to be updated for Streamlit Cloud."
    )
    with st.expander("Technical details"):
        st.exception(e)
    st.stop()


# -----------------------------
# Constants
# -----------------------------
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "outputs" / "model_bundle.joblib"
DEFAULT_MEDIUM_THRESHOLD = 0.10  # demo threshold
MAX_UPLOAD_BYTES = 2_000_000  # 2 MB


# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def load_bundle(bundle_path: Path) -> dict[str, Any]:
    return joblib.load(bundle_path)


def risk_band(prob: float, thr_medium: float, thr_high: float) -> str:
    if prob < thr_medium:
        return "LOW"
    if prob < thr_high:
        return "MEDIUM"
    return "HIGH"


def band_badge(band: str) -> str:
    # Small, non-tacky, compact
    if band == "LOW":
        return "🟢 LOW"
    if band == "MEDIUM":
        return "🟠 MEDIUM"
    return "🔴 HIGH"


def friendly_feature_name(feat: str) -> str:
    """
    Convert model feature names like 'RespRate_last' into clinician-ish labels.
    Unknown features fall back to raw name.
    """
    base_map = {
        "HeartRate": "Heart rate",
        "SysBP": "Systolic BP",
        "DiasBP": "Diastolic BP",
        "MeanBP": "Mean arterial pressure",
        "RespRate": "Respiratory rate",
        "Temp": "Temperature",
        "SaO2": "Oxygen saturation (SaO₂)",
        "FiO2": "FiO₂",
        "pH": "pH",
        "Glucose": "Glucose",
        "Lactate": "Lactate",
        "BUN": "Urea (BUN)",
        "Creatinine": "Creatinine",
        "HCO3": "Bicarbonate (HCO₃⁻)",
        "Hct": "Haematocrit",
        "WBC": "WBC",
        "Platelets": "Platelets",
        "Na": "Sodium (Na)",
        "K": "Potassium (K)",
        "Bilirubin": "Bilirubin",
        "Age": "Age",
        "MechVent": "Mechanical ventilation",
    }

    suffix_map = {
        "_mean": " (average)",
        "_last": " (latest)",
        "_was_measured": " (measured?)",
        "_prop_on": " (proportion of time on ventilator)",
    }

    for suf, label in suffix_map.items():
        if feat.endswith(suf):
            base = feat[: -len(suf)]
            return f"{base_map.get(base, base)}{label}"

    return base_map.get(feat, feat)


def safe_float(v: Any) -> float:
    """
    Parse a value into float; returns np.nan if not parseable.
    """
    if v is None:
        return np.nan
    if v is pd.NA:
        return np.nan
    if isinstance(v, (float, int, np.floating, np.integer)):
        return float(v)
    s = str(v).strip()
    if s == "":
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def parse_measurements(text: str) -> list[float]:
    """
    Parse comma/space-separated measurements.
    Example: "12, 14, 16" or "12 14 16"
    Returns floats; ignores blanks/unparseable tokens.
    """
    if text is None:
        return []
    s = str(text).strip()
    if not s:
        return []
    # Replace common separators with commas, then split
    s = s.replace(";", ",").replace("\n", ",").replace("\t", ",").replace(" ", ",")
    tokens = [t.strip() for t in s.split(",") if t.strip() != ""]
    vals: list[float] = []
    for t in tokens:
        # allow % like "60%" for FiO2 entry convenience
        t2 = t.replace("%", "").strip()
        try:
            vals.append(float(t2))
        except Exception:
            continue
    return vals


def build_engineered_from_manual(
    starter_vars: list[str],
    inputs: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """
    Build the engineered features dict expected by the model from clinician manual inputs.

    We fill:
      {Var}_was_measured, {Var}_mean, {Var}_last
    Plus special:
      MechVent_prop_on, MechVent_last (numeric 0/1), FiO2 scaling (percent->fraction)

    Returns:
      feats_dict, warnings_list (clinician-readable)
    """
    feats: dict[str, Any] = {}
    warnings: list[str] = []

    for var in starter_vars:
        key = var  # matches training feature prefixes
        raw = inputs.get(var, "")

        # Special: Age is usually a single value
        if var.lower() == "age":
            age = safe_float(raw)
            if np.isnan(age):
                feats[f"{key}_was_measured"] = 0
                feats[f"{key}_mean"] = np.nan
                feats[f"{key}_last"] = np.nan
            else:
                feats[f"{key}_was_measured"] = 1
                feats[f"{key}_mean"] = float(age)
                feats[f"{key}_last"] = float(age)
            continue

        # Special: MechVent is Yes/No/Unknown
        if var.lower() == "mechvent":
            mv = str(raw).strip().lower()
            if mv in ("yes", "y", "1", "true"):
                vals = [1.0]
            elif mv in ("no", "n", "0", "false"):
                vals = [0.0]
            else:
                vals = []

            if not vals:
                feats[f"{key}_was_measured"] = 0
                feats[f"{key}_mean"] = np.nan
                feats[f"{key}_last"] = np.nan
                feats["MechVent_prop_on"] = np.nan
                feats["MechVent_last"] = np.nan
            else:
                feats[f"{key}_was_measured"] = 1
                feats[f"{key}_mean"] = float(np.mean(vals))
                feats[f"{key}_last"] = float(vals[-1])
                feats["MechVent_prop_on"] = float(np.mean(vals))
                feats["MechVent_last"] = float(vals[-1])
            continue

        # General numeric vars: allow multiple measurements (comma/space separated)
        vals = parse_measurements(str(raw))

        if len(vals) == 0:
            feats[f"{key}_was_measured"] = 0
            feats[f"{key}_mean"] = np.nan
            feats[f"{key}_last"] = np.nan
            continue

        # FiO2 special handling: if clinician enters 40, 60 etc treat as percent
        if var.lower() == "fio2":
            vmax = max(vals) if vals else np.nan
            if not np.isnan(vmax) and vmax > 1.5:
                vals = [v / 100.0 for v in vals]

        feats[f"{key}_was_measured"] = 1
        feats[f"{key}_mean"] = float(np.mean(vals))
        feats[f"{key}_last"] = float(vals[-1])

    return feats, warnings


def align_and_impute(
    feats: dict[str, Any],
    feature_columns: list[str],
    imputer,
) -> tuple[pd.DataFrame, int, list[str], pd.DataFrame]:
    """
    Align engineered features to the training feature set and run the saved imputer.

    CRITICAL: prevent pd.NA from reaching sklearn (causes NAType crash).
    """
    X = pd.DataFrame([feats])

    # Ensure all expected columns exist (fill missing with np.nan, NOT pd.NA)
    for c in feature_columns:
        if c not in X.columns:
            X[c] = np.nan

    # Keep only training columns, in correct order
    X = X[feature_columns]

    # Kill pd.NA everywhere (the core fix for NAType crashes)
    X = X.replace({pd.NA: np.nan})

    # Coerce to numeric (strings -> NaN)
    X = X.apply(pd.to_numeric, errors="coerce")

    # Force float dtype so sklearn never sees object/NAType
    X = X.astype(float)

    present_mask = ~X.isna().iloc[0]
    present_count = int(present_mask.sum())
    missing_features = [col for col, ok in zip(feature_columns, present_mask) if not ok]

    # Feed numpy array into sklearn (extra safety)
    X_imp_arr = imputer.transform(X.to_numpy(dtype=float))
    X_imp = pd.DataFrame(X_imp_arr, columns=feature_columns)

    return X_imp, present_count, missing_features, X


def show_friendly_error(msg: str, debug_exc: Exception | None = None) -> None:
    st.error(msg)
    if debug_exc is not None:
        with st.expander("Technical details (for the developer / portfolio)"):
            st.exception(debug_exc)


# -----------------------------
# UI Header
# -----------------------------
st.title("ICU Mortality Risk Demo")
st.caption("Educational demo only — not for clinical decision-making.")

with st.expander("About this demo"):
    st.write(
        """
This tool shows how a prediction service *could* work end-to-end:
it reads clinical measurements, prepares them in the same format the model was trained on,
and returns an estimated risk score.

**What it is:** a portfolio demo using a model trained on a public ICU dataset.  
**What it isn’t:** a validated clinical device, and it has not been evaluated for real-world safety or performance.
        """.strip()
    )


# -----------------------------
# Load model bundle
# -----------------------------
if not DEFAULT_BUNDLE_PATH.exists():
    show_friendly_error(
        "The prediction model isn't available in this deployment. "
        "The repository should include `outputs/model_bundle.joblib`."
    )
    st.stop()

bundle = load_bundle(DEFAULT_BUNDLE_PATH)

# Required bundle parts
try:
    model = bundle["model"]
    imputer = bundle["imputer"]
    feature_columns = bundle["feature_columns"]
except Exception as e:
    show_friendly_error("The model bundle looks incomplete or corrupted.", e)
    st.stop()

thr_high = safe_float(bundle.get("default_threshold", 0.5))
if np.isnan(thr_high):
    thr_high = 0.5

thr_medium = safe_float(bundle.get("medium_threshold", DEFAULT_MEDIUM_THRESHOLD))
if np.isnan(thr_medium):
    thr_medium = DEFAULT_MEDIUM_THRESHOLD

starter_vars = bundle.get("starter_vars", None)
if not isinstance(starter_vars, list) or len(starter_vars) == 0:
    # Fallback: infer base vars from columns containing _mean/_last/_was_measured
    bases = set()
    for c in feature_columns:
        for suf in ("_mean", "_last", "_was_measured"):
            if c.endswith(suf):
                bases.add(c[: -len(suf)])
    starter_vars = sorted(bases)

# -----------------------------
# Mode selector
# -----------------------------
mode = st.radio(
    "Choose input method",
    ["Upload patient file (.txt)", "Manual entry (quick)"],
    horizontal=True,
)

# Compact legend (shows all bands without huge text)
st.markdown(
    f"**Risk bands (demo thresholds):** "
    f"🟢 LOW < `{thr_medium:.3f}`   •   "
    f"🟠 MEDIUM `{thr_medium:.3f}` to `< {thr_high:.3f}`   •   "
    f"🔴 HIGH ≥ `{thr_high:.3f}`"
)

st.divider()


# -----------------------------
# Upload Mode
# -----------------------------
if mode == "Upload patient file (.txt)":
    uploaded = st.file_uploader("Upload a single patient `.txt` file", type=["txt"])

    if uploaded is None:
        st.info("Upload a file to get a prediction.")
        st.stop()

    if uploaded.size > MAX_UPLOAD_BYTES:
        st.error("That file is too large for this demo (max 2MB).")
        st.stop()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = Path(tmp.name)

        long_df = load_patient_long(tmp_path)
        feats = summarise_patient(long_df)

        X_imp, present_count, missing_features, X_aligned = align_and_impute(feats, feature_columns, imputer)

        prob = float(model.predict_proba(X_imp)[:, 1][0])
        band = risk_band(prob, thr_medium, thr_high)

        st.subheader("Result")
        c1, c2, c3 = st.columns([1.2, 1, 1])
        c1.metric("Predicted risk", f"{prob*100:.2f}%")
        c2.metric("Risk band", band_badge(band))
        c3.metric("Data completeness", f"{present_count}/{len(feature_columns)}")

        st.progress(min(max(prob, 0.0), 1.0))

        with st.expander("Data completeness details"):
            missing_count = len(missing_features)
            st.write(f"Missing inputs used by the model: **{missing_count}** (handled by imputation).")
            if missing_count:
                nice = [friendly_feature_name(x) for x in missing_features]
                st.write(nice)

        with st.expander("Show engineered features (non-empty only)"):
            # Show only features that are actually present (not NaN)
            row = X_aligned.iloc[0]
            non_empty = row[~row.isna()]
            df_show = pd.DataFrame({"feature": non_empty.index, "value": non_empty.values})
            # Friendly labels
            df_show["feature"] = df_show["feature"].map(friendly_feature_name)
            st.dataframe(df_show, use_container_width=True)

    except Exception as e:
        show_friendly_error(
            "Something went wrong while predicting from the uploaded file. "
            "For this demo, the file must match the expected ICU dataset format.",
            e,
        )
    finally:
        try:
            tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass

    st.stop()


# -----------------------------
# Manual Mode (Quick)
# -----------------------------
st.subheader("Manual entry (quick)")
st.caption(
    "Enter **multiple measurements** as comma/space-separated values (oldest → newest). "
    "If you only have one measurement, enter one value."
)

# A clinician-friendly spec for common ICU variables
VAR_SPECS: dict[str, dict[str, str]] = {
    "Age": {"label": "Age", "unit": "years", "hint": "Example: 67"},
    "HeartRate": {"label": "Heart rate", "unit": "bpm", "hint": "Example: 96 102 110"},
    "SysBP": {"label": "Systolic BP", "unit": "mmHg", "hint": "Example: 110, 98, 105"},
    "DiasBP": {"label": "Diastolic BP", "unit": "mmHg", "hint": "Example: 60 55 58"},
    "MeanBP": {"label": "Mean arterial pressure (MAP)", "unit": "mmHg", "hint": "Example: 75 65 70"},
    "RespRate": {"label": "Respiratory rate", "unit": "/min", "hint": "Example: 18 22 26"},
    "Temp": {"label": "Temperature", "unit": "°C", "hint": "Example: 36.8 37.2"},
    "SaO2": {"label": "Oxygen saturation (SaO₂)", "unit": "%", "hint": "Example: 95 92 90"},
    "FiO2": {"label": "FiO₂", "unit": "fraction or %", "hint": "Example: 0.4 0.6  OR  40 60"},
    "pH": {"label": "pH", "unit": "", "hint": "Example: 7.36 7.31"},
    "Glucose": {"label": "Glucose", "unit": "mmol/L (or dataset units)", "hint": "Example: 7.2 8.1 9.0"},
    "Lactate": {"label": "Lactate", "unit": "mmol/L", "hint": "Example: 1.8 2.4 3.1"},
    "BUN": {"label": "Urea (BUN)", "unit": "mmol/L (or dataset units)", "hint": "Example: 6 10 14"},
    "Creatinine": {"label": "Creatinine", "unit": "µmol/L (or dataset units)", "hint": "Example: 90 120"},
    "HCO3": {"label": "Bicarbonate (HCO₃⁻)", "unit": "mmol/L", "hint": "Example: 24 21 19"},
    "Hct": {"label": "Haematocrit", "unit": "%", "hint": "Example: 40 36"},
    "WBC": {"label": "WBC", "unit": "x10⁹/L", "hint": "Example: 7.0 12.5"},
    "Platelets": {"label": "Platelets", "unit": "x10⁹/L", "hint": "Example: 250 180"},
    "Na": {"label": "Sodium (Na)", "unit": "mmol/L", "hint": "Example: 138 132"},
    "K": {"label": "Potassium (K)", "unit": "mmol/L", "hint": "Example: 4.1 5.0"},
    "Bilirubin": {"label": "Bilirubin", "unit": "µmol/L", "hint": "Example: 10 35"},
    "MechVent": {"label": "Mechanical ventilation (current)", "unit": "", "hint": "Choose Yes/No"},
}

# Show only variables the model expects (starter_vars)
starter_vars = [v for v in starter_vars if isinstance(v, str)]

with st.form("manual_form", clear_on_submit=False):
    st.write("### Core measurements")

    inputs: dict[str, Any] = {}
    parse_messages: list[str] = []

    # Two-column layout for nicer clinician UX
    cols = st.columns(2)
    col_i = 0

    for var in starter_vars:
        spec = VAR_SPECS.get(var, {"label": var, "unit": "", "hint": "Enter values like: 1 2 3"})

        with cols[col_i]:
            if var.lower() == "mechvent":
                mv = st.selectbox(
                    f"{spec['label']}",
                    ["Unknown", "Yes", "No"],
                    index=0,
                    help="Whether the patient is mechanically ventilated at the time of measurement.",
                )
                inputs[var] = mv
            else:
                unit = f" ({spec['unit']})" if spec.get("unit") else ""
                inputs[var] = st.text_input(
                    f"{spec['label']}{unit}",
                    value="",
                    placeholder=spec.get("hint", ""),
                    help="You can paste multiple values separated by spaces or commas (oldest → newest).",
                )

        col_i = 1 - col_i  # toggle columns

    st.write("---")
    submitted = st.form_submit_button("Calculate risk")

if not submitted:
    st.stop()

try:
    # Convert selectbox strings into what our builder expects for MechVent
    if "MechVent" in inputs:
        mv = str(inputs["MechVent"]).strip().lower()
        if mv == "yes":
            inputs["MechVent"] = "yes"
        elif mv == "no":
            inputs["MechVent"] = "no"
        else:
            inputs["MechVent"] = ""

    raw_engineered, warnings = build_engineered_from_manual(starter_vars, inputs)

    X_imp, present_count, missing_features, X_aligned = align_and_impute(raw_engineered, feature_columns, imputer)

    prob = float(model.predict_proba(X_imp)[:, 1][0])
    band = risk_band(prob, thr_medium, thr_high)

    st.subheader("Result")
    c1, c2, c3 = st.columns([1.2, 1, 1])
    c1.metric("Predicted risk", f"{prob*100:.2f}%")
    c2.metric("Risk band", band_badge(band))
    c3.metric("Data completeness", f"{present_count}/{len(feature_columns)}")
    st.progress(min(max(prob, 0.0), 1.0))

    # Clinician-friendly input issues
    # (We don't throw scary parsing errors; we explain missing handling calmly.)
    st.subheader("Input check")
    st.write(
        "If anything was left blank or unreadable, it was treated as **missing** "
        "and handled by the model’s built-in missing-data strategy."
    )

    missing_count = len(missing_features)
    st.write(f"**Missing model inputs (handled automatically):** {missing_count}")

    if missing_count:
        with st.expander("Show which items were missing"):
            nice = [friendly_feature_name(x) for x in missing_features]
            st.write(nice)

    with st.expander("Show what the model actually received (non-empty engineered features)"):
        row = X_aligned.iloc[0]
        non_empty = row[~row.isna()]
        df_show = pd.DataFrame({"feature": non_empty.index, "value": non_empty.values})
        df_show["feature"] = df_show["feature"].map(friendly_feature_name)
        st.dataframe(df_show, use_container_width=True)

except Exception as e:
    # Never show clinicians raw stack traces outside the expander
    show_friendly_error(
        "Something went wrong while calculating risk from manual entry. "
        "Double-check that numbers are entered like `12` or `12.5` (and for multiple measurements: `12 14 16`).",
        e,
    )
