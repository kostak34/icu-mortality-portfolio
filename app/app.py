"""
Streamlit demo app (educational).

Run locally:
  python -m pip install -r requirements.txt
  python -m streamlit run app/app.py

Notes:
- Requires outputs/model_bundle.joblib committed in the repo for Streamlit Cloud
- Upload mode: parse patient .txt → features → impute → predict
- Manual mode: clinician enters values at 12h/6h/now → engineer mean/last → impute → predict
- Educational demo only. Not for clinical use.
"""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
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


# -----------------------------
# Constants / helpers
# -----------------------------
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "outputs" / "model_bundle.joblib"
DEFAULT_LOW_THRESHOLD = 0.10
MAX_UPLOAD_BYTES = 2_000_000  # 2MB

TIMEPOINTS = [
    ("12 hours ago", "t12"),
    ("6 hours ago", "t6"),
    ("Now", "t0"),
]

RISK_COLOR = {"LOW": "🟢", "MEDIUM": "🟠", "HIGH": "🔴"}

# Clinician-friendly labels + units + plausible ranges
VAR_META: dict[str, dict[str, Any]] = {
    "HR": {"label": "Heart rate", "units": "bpm", "min": 0.0, "max": 250.0, "type": "num"},
    "RespRate": {"label": "Respiratory rate", "units": "breaths/min", "min": 0.0, "max": 80.0, "type": "num"},
    "SysBP": {"label": "Systolic BP", "units": "mmHg", "min": 0.0, "max": 300.0, "type": "num"},
    "DiasBP": {"label": "Diastolic BP", "units": "mmHg", "min": 0.0, "max": 200.0, "type": "num"},
    "MeanBP": {"label": "MAP", "units": "mmHg", "min": 0.0, "max": 200.0, "type": "num"},
    "Temp": {"label": "Temperature", "units": "°C", "min": 25.0, "max": 45.0, "type": "num"},
    "SaO2": {"label": "Oxygen saturation", "units": "%", "min": 0.0, "max": 100.0, "type": "num"},
    "FiO2": {"label": "FiO₂", "units": "% (21–100)", "min": 21.0, "max": 100.0, "type": "fio2_percent"},
    "MechVent": {"label": "Invasive mechanical ventilation", "units": "Yes/No", "type": "binary"},
    "pH": {"label": "pH", "units": "", "min": 6.6, "max": 7.8, "type": "num"},
    "Lactate": {"label": "Lactate", "units": "mmol/L", "min": 0.0, "max": 30.0, "type": "num"},
    "Glucose": {"label": "Glucose", "units": "mmol/L", "min": 0.0, "max": 60.0, "type": "num"},
    "Creatinine": {"label": "Creatinine", "units": "µmol/L", "min": 0.0, "max": 2000.0, "type": "num"},
    "BUN": {"label": "Urea (BUN)", "units": "mmol/L", "min": 0.0, "max": 80.0, "type": "num"},
    "WBC": {"label": "White cell count", "units": "x10⁹/L", "min": 0.0, "max": 200.0, "type": "num"},
    "Platelets": {"label": "Platelets", "units": "x10⁹/L", "min": 0.0, "max": 2000.0, "type": "num"},
}


@st.cache_resource
def load_bundle(bundle_path: Path) -> dict:
    return joblib.load(bundle_path)


def safe_float(x: Any) -> float | None:
    if x is None or x is pd.NA:
        return None
    try:
        if isinstance(x, str) and x.strip() == "":
            return None
        return float(x)
    except Exception:
        return None


def humanise_feature_name(name: str) -> str:
    """
    Convert model feature names like `RespRate_mean` into clinician-friendly labels.
    Falls back gracefully if the variable isn't in VAR_META.
    """
    # Special case
    if name == "MechVent_prop_on":
        base = VAR_META.get("MechVent", {}).get("label", "Mechanical ventilation")
        return f"{base} (proportion on)"

    # Pattern: <Var>_<suffix>
    if "_" in name:
        base, suffix = name.split("_", 1)
        base_label = VAR_META.get(base, {}).get("label", base)
        suffix_map = {
            "mean": "mean",
            "last": "most recent",
            "was": None,  # handled below if ever seen
        }
        if suffix == "was_measured":
            return f"{base_label} (recorded: yes/no)"
        if suffix in suffix_map and suffix_map[suffix] is not None:
            return f"{base_label} ({suffix_map[suffix]})"
        # Unknown suffix
        return f"{base_label} ({suffix.replace('_', ' ')})"

    # No underscore
    return VAR_META.get(name, {}).get("label", name)


def align_and_impute(
    feats: dict,
    feature_columns: list[str],
    imputer,
) -> tuple[pd.DataFrame, int, list[str], pd.DataFrame]:
    X = pd.DataFrame([feats])

    for c in feature_columns:
        if c not in X.columns:
            X[c] = np.nan

    X = X[feature_columns]

    missing_mask = X.isna().iloc[0]
    missing_features = X.columns[missing_mask].tolist()
    present_count = int((~missing_mask).sum())

    X = X.replace({pd.NA: np.nan})
    X = X.where(pd.notna(X), np.nan)
    X = X.apply(pd.to_numeric, errors="coerce").astype("float64")

    X_imp = pd.DataFrame(imputer.transform(X), columns=feature_columns)
    return X_imp, present_count, missing_features, X


def risk_band(prob: float, thr_low: float, thr_high: float) -> str:
    if prob < thr_low:
        return "LOW"
    if prob < thr_high:
        return "MEDIUM"
    return "HIGH"


def show_risk_banner(band: str) -> None:
    icon = RISK_COLOR[band]
    if band == "LOW":
        st.success(f"{icon} **LOW RISK**")
    elif band == "MEDIUM":
        st.warning(f"{icon} **MEDIUM RISK**")
    else:
        st.error(f"{icon} **HIGH RISK**")


def normalise_fio2_percent_to_fraction(v: float | None) -> float | None:
    if v is None:
        return None
    return float(v) / 100.0


def manual_inputs_to_engineered_features(
    inputs: dict[str, dict[str, Any]],
    starter_vars: list[str],
) -> dict:
    feat: dict[str, Any] = {}

    def latest_value(t0, t6, t12):
        for v in [t0, t6, t12]:
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return v
        return None

    for var in starter_vars:
        meta = VAR_META.get(var, {"label": var, "units": "", "min": -1e6, "max": 1e6, "type": "num"})
        v12_raw = inputs.get(var, {}).get("t12", None)
        v6_raw = inputs.get(var, {}).get("t6", None)
        v0_raw = inputs.get(var, {}).get("t0", None)

        if meta.get("type") == "binary":
            def yn_to_num(x: Any) -> float | None:
                if x == "Yes":
                    return 1.0
                if x == "No":
                    return 0.0
                return None

            v12 = yn_to_num(v12_raw)
            v6 = yn_to_num(v6_raw)
            v0 = yn_to_num(v0_raw)

            series = [v for v in [v12, v6, v0] if v is not None]
            feat[f"{var}_was_measured"] = 1 if len(series) > 0 else 0

            mean_val = float(np.mean(series)) if len(series) > 0 else np.nan
            last_val = latest_value(v0, v6, v12)
            last_val = float(last_val) if last_val is not None else np.nan

            feat[f"{var}_mean"] = mean_val
            feat[f"{var}_last"] = last_val

            if var.lower() == "mechvent":
                feat["MechVent_prop_on"] = mean_val if not np.isnan(mean_val) else np.nan
                feat["MechVent_last"] = last_val if not np.isnan(last_val) else np.nan

            continue

        v12 = safe_float(v12_raw)
        v6 = safe_float(v6_raw)
        v0 = safe_float(v0_raw)

        if meta.get("type") == "fio2_percent":
            v12 = normalise_fio2_percent_to_fraction(v12)
            v6 = normalise_fio2_percent_to_fraction(v6)
            v0 = normalise_fio2_percent_to_fraction(v0)

        series = [v for v in [v12, v6, v0] if v is not None]
        feat[f"{var}_was_measured"] = 1 if len(series) > 0 else 0

        if len(series) == 0:
            feat[f"{var}_mean"] = np.nan
            feat[f"{var}_last"] = np.nan
        else:
            feat[f"{var}_mean"] = float(np.mean(series))
            last_val = latest_value(v0, v6, v12)
            feat[f"{var}_last"] = float(last_val) if last_val is not None else np.nan

    return feat


def count_variables_entered(inputs: dict[str, dict[str, Any]], starter_vars: list[str]) -> int:
    count = 0
    for var in starter_vars:
        meta = VAR_META.get(var, {"type": "num"})
        vtype = meta.get("type", "num")

        t12 = inputs.get(var, {}).get("t12", None)
        t6 = inputs.get(var, {}).get("t6", None)
        t0 = inputs.get(var, {}).get("t0", None)

        if vtype == "binary":
            provided = any(v in ("Yes", "No") for v in [t12, t6, t0])
        else:
            provided = any(v is not None for v in [t12, t6, t0])

        if provided:
            count += 1
    return count


def count_numeric_engineered_present(raw_engineered: dict, starter_vars: list[str]) -> tuple[int, int]:
    present = 0
    total = 0
    for var in starter_vars:
        for suffix in ["_mean", "_last"]:
            k = f"{var}{suffix}"
            total += 1
            v = raw_engineered.get(k, np.nan)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                present += 1
    if any(v.lower() == "mechvent" for v in starter_vars):
        total += 1
        v = raw_engineered.get("MechVent_prop_on", np.nan)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            present += 1
    return present, total


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="ICU Mortality Demo", layout="centered")

st.title("ICU Mortality Prediction (Demo)")
st.caption("Educational demo only. Not for clinical use.")

with st.expander("What this is (and isn’t)"):
    st.write(
        """
        This demo estimates mortality risk from routine ICU observations and blood results.
        It’s designed to show how a decision-support tool *could* work in principle.

        It has **not** been validated for real-world clinical use and must not be used for patient care.
        """
    )

bundle_path = DEFAULT_BUNDLE_PATH
if not bundle_path.exists():
    st.error("Model bundle not found. Expected: outputs/model_bundle.joblib")
    st.stop()

bundle = load_bundle(bundle_path)
model = bundle["model"]
imputer = bundle["imputer"]
feature_columns = bundle["feature_columns"]

thr_high = float(bundle.get("default_threshold", 0.5))
thr_low = float(bundle.get("low_threshold", DEFAULT_LOW_THRESHOLD))
starter_vars = bundle.get("starter_vars", None)

st.subheader("Risk bands (demo cut-offs)")
c1, c2, c3 = st.columns(3)
c1.metric("LOW risk", f"< {thr_low:.3f}")
c2.metric("MEDIUM risk", f"{thr_low:.3f} to {thr_high:.3f}")
c3.metric("HIGH risk", f"≥ {thr_high:.3f}")

st.divider()

if "page" not in st.session_state:
    st.session_state.page = "Upload patient file (.txt)"

page = st.radio(
    "Mode",
    ["Upload patient file (.txt)", "Manual entry"],
    key="page",
    horizontal=True,
    label_visibility="collapsed",
)

# -----------------------------
# Upload mode
# -----------------------------
if page == "Upload patient file (.txt)":
    uploaded = st.file_uploader("Upload a single patient .txt file", type=["txt"])

    if uploaded is None:
        st.info("Upload a patient file to generate a prediction.")
    else:
        if uploaded.size > MAX_UPLOAD_BYTES:
            st.error("File too large for this demo (max 2MB).")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = Path(tmp.name)

            try:
                long_df = load_patient_long(tmp_path)
                feats = summarise_patient(long_df)

                X_imp, present_count, missing_features, _ = align_and_impute(feats, feature_columns, imputer)

                prob = float(model.predict_proba(X_imp)[:, 1][0])
                band = risk_band(prob, thr_low, thr_high)

                st.subheader("Result")
                show_risk_banner(band)
                st.metric("Predicted mortality risk (probability)", f"{prob:.6f}")

                st.subheader("Data completeness")
                st.write(f"Model features missing (imputed): **{len(missing_features)}**")
                if missing_features:
                    with st.expander("Show missing data items"):
                        st.write([humanise_feature_name(x) for x in missing_features])

            except Exception:
                st.error("Something went wrong while predicting from the uploaded file.")
                st.info("Tip: try a different patient file. This demo expects the same format used during training.")
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

# -----------------------------
# Manual mode
# -----------------------------
else:
    if starter_vars is None or not isinstance(starter_vars, list) or len(starter_vars) == 0:
        st.warning("Manual entry is unavailable because this model bundle does not include `starter_vars`.")
    else:
        st.write(
            """
            Enter values at **12 hours ago**, **6 hours ago**, and **Now**.
            The tool converts these into the model inputs (average + most recent value).
            """
        )

        GROUPS = {
            "Respiratory & ventilation": ["RespRate", "SaO2", "FiO2", "MechVent"],
            "Haemodynamics": ["HR", "SysBP", "DiasBP", "MeanBP", "Temp"],
            "Blood gas / perfusion": ["pH", "Lactate"],
            "Bloods / labs": ["Glucose", "Creatinine", "BUN", "WBC", "Platelets"],
        }

        listed = {v for g in GROUPS.values() for v in g}
        extras = [v for v in starter_vars if v not in listed]

        inputs: dict[str, dict[str, Any]] = {v: {} for v in starter_vars}

        submitted = False
        with st.form("manual_form", clear_on_submit=False):
            for group_name, var_list in GROUPS.items():
                vars_in_group = [v for v in var_list if v in starter_vars]
                if not vars_in_group:
                    continue

                st.markdown(f"## {group_name}")

                for var in vars_in_group:
                    meta = VAR_META.get(var, {"label": var, "units": "", "min": -1e6, "max": 1e6, "type": "num"})
                    label = meta.get("label", var)
                    units = meta.get("units", "")
                    vtype = meta.get("type", "num")

                    st.markdown(f"**{label}**" + (f" ({units})" if units else ""))

                    cols = st.columns(3)
                    for idx, (tp_label, tp_key) in enumerate(TIMEPOINTS):
                        if vtype == "binary":
                            inputs[var][tp_key] = cols[idx].selectbox(
                                tp_label,
                                options=["Missing", "Yes", "No"],
                                index=0,
                                key=f"{var}_{tp_key}_bin",
                                help="Choose Yes / No. Leave as Missing if unknown.",
                            )
                        else:
                            min_v = float(meta.get("min", -1e6))
                            max_v = float(meta.get("max", 1e6))
                            help_text = (
                                "Enter FiO₂ as a percentage (21–100)."
                                if vtype == "fio2_percent"
                                else f"Expected range: {min_v:g}–{max_v:g}" + (f" {units}" if units else "")
                            )

                            inputs[var][tp_key] = cols[idx].number_input(
                                tp_label,
                                min_value=min_v,
                                max_value=max_v,
                                value=None,
                                step=1.0 if (max_v - min_v) > 50 else 0.1,
                                key=f"{var}_{tp_key}_num",
                                help=help_text,
                                format="%.3f" if (max_v - min_v) <= 20 else "%.1f",
                            )

                    st.write("")

                st.divider()

            if extras:
                st.markdown("## Other variables")
                for var in extras:
                    meta = VAR_META.get(var, {"label": var, "units": "", "min": -1e6, "max": 1e6, "type": "num"})
                    label = meta.get("label", var)
                    units = meta.get("units", "")
                    vtype = meta.get("type", "num")

                    st.markdown(f"**{label}**" + (f" ({units})" if units else ""))
                    cols = st.columns(3)
                    for idx, (tp_label, tp_key) in enumerate(TIMEPOINTS):
                        if vtype == "binary":
                            inputs[var][tp_key] = cols[idx].selectbox(
                                tp_label,
                                options=["Missing", "Yes", "No"],
                                index=0,
                                key=f"{var}_{tp_key}_bin_extra",
                            )
                        else:
                            min_v = float(meta.get("min", -1e6))
                            max_v = float(meta.get("max", 1e6))
                            inputs[var][tp_key] = cols[idx].number_input(
                                tp_label,
                                min_value=min_v,
                                max_value=max_v,
                                value=None,
                                step=1.0 if (max_v - min_v) > 50 else 0.1,
                                key=f"{var}_{tp_key}_num_extra",
                            )
                    st.write("")

                st.divider()

            submitted = st.form_submit_button("Calculate risk")

        if submitted:
            try:
                entered_vars = count_variables_entered(inputs, starter_vars)

                raw_engineered = manual_inputs_to_engineered_features(inputs, starter_vars)
                numeric_present, numeric_total = count_numeric_engineered_present(raw_engineered, starter_vars)

                X_imp, _present_count, missing_features, _ = align_and_impute(raw_engineered, feature_columns, imputer)

                prob = float(model.predict_proba(X_imp)[:, 1][0])
                band = risk_band(prob, thr_low, thr_high)

                st.subheader("Result")
                show_risk_banner(band)
                st.metric("Predicted mortality risk (probability)", f"{prob:.6f}")

                st.subheader("What was actually entered")
                c1, c2 = st.columns(2)
                c1.metric("Variables entered", f"{entered_vars}/{len(starter_vars)}")
                c2.metric("Engineered values available", f"{numeric_present}/{numeric_total}")

                st.subheader("Missing data items")
                st.write(f"Missing items used by the model (imputed): **{len(missing_features)}**")
                if missing_features:
                    with st.expander("Show missing items"):
                        st.write([humanise_feature_name(x) for x in missing_features])

            except Exception:
                st.error("Something went wrong while predicting from manual entry.")
                st.info("Please check entries and try again.")

# -----------------------------
# Technical details (bottom)
# -----------------------------
with st.expander("Technical details (for reviewers / developers)"):
    st.write(f"Bundle path: `{bundle_path.as_posix()}`")
    st.write(f"Features expected: {len(feature_columns)}")
    st.write(f"Starter vars in bundle: {len(starter_vars) if isinstance(starter_vars, list) else 'missing'}")
