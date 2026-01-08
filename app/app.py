# app/app.py
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Robust import setup (works locally + on Streamlit Cloud)
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Add BOTH the project root (for `import scripts...`) and scripts dir
# (so older absolute imports inside scripts like `import step_01_load_raw` still work)
for p in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Your existing functions
from scripts.step_01_load_raw import load_patient_long  # type: ignore
from scripts.step_02_batch_features import summarise_patient  # type: ignore

DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "outputs" / "model_bundle.joblib"
DEFAULT_MEDIUM_THRESHOLD = 0.10  # demo medium-risk threshold


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
@st.cache_resource
def load_bundle(bundle_path: Path) -> dict:
    return joblib.load(bundle_path)


def risk_band(prob: float, thr_medium: float, thr_high: float) -> str:
    if prob < thr_medium:
        return "LOW"
    if prob < thr_high:
        return "MEDIUM"
    return "HIGH"


def render_risk_pill(band: str) -> None:
    # compact, non-tacky badge
    styles = {
        "LOW":    ("#0f5132", "#d1e7dd"),
        "MEDIUM": ("#664d03", "#fff3cd"),
        "HIGH":   ("#842029", "#f8d7da"),
    }
    fg, bg = styles.get(band, ("#0b0f14", "#e9ecef"))
    st.markdown(
        f"""
        <div style="
            display:inline-block;
            padding:6px 10px;
            border-radius:999px;
            background:{bg};
            color:{fg};
            font-weight:600;
            font-size:0.95rem;
            line-height:1;
        ">
            Risk band: {band}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _coerce_numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert any pd.NA/None/object → np.nan and coerce to numeric.
    This is the critical part that prevents NAType crashes in sklearn.
    """
    X = X.copy()
    X = X.where(pd.notna(X), np.nan)
    # coerce everything to numeric; non-numeric becomes NaN
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    # ensure float dtype for sklearn
    return X.astype(float)


def align_and_impute(
    feats: dict,
    feature_columns: List[str],
    imputer: Any,
) -> Tuple[pd.DataFrame, int, List[str], pd.DataFrame]:
    """
    Align engineered feature dict to the training columns and impute.
    Returns:
      X_imp, present_count, missing_features, X_aligned (pre-imputation, aligned)
    """
    X = pd.DataFrame([feats])

    # Add any missing expected columns as NaN
    for c in feature_columns:
        if c not in X.columns:
            X[c] = np.nan

    # Drop any extras and keep training order
    X = X[feature_columns]

    missing_mask = X.isna().iloc[0]
    missing_features = X.columns[missing_mask].tolist()
    present_count = int((~missing_mask).sum())

    X_num = _coerce_numeric_frame(X)

    # Impute using the fitted training imputer
    X_imp_arr = imputer.transform(X_num)
    X_imp = pd.DataFrame(X_imp_arr, columns=feature_columns)

    return X_imp, present_count, missing_features, X


# ------------------------------------------------------------
# Manual entry schema + parsing
# ------------------------------------------------------------
def parse_optional_float(text: str) -> Tuple[float | None, str | None]:
    """
    Returns (value or None, error_message or None)
    Empty -> None
    """
    if text is None:
        return None, None
    s = str(text).strip()
    if s == "":
        return None, None
    try:
        return float(s), None
    except Exception:
        return None, "Please enter a number (e.g., 6.2) or leave blank if unknown."


def validate_range(
    var_label: str,
    value: float,
    lo: float,
    hi: float,
    units: str,
) -> str | None:
    if value < lo or value > hi:
        return f"{var_label}: please enter a value between {lo:g} and {hi:g}{(' ' + units) if units else ''}."
    return None


def fio2_to_fraction(value: float) -> float | None:
    """
    Accept either fraction (0.21–1.0) or percent (21–100).
    Convert percent → fraction.
    """
    if value is None:
        return None
    if value > 1.5:  # treat as %
        return value / 100.0
    return value


def last_from_timepoints(v12: float | None, v6: float | None, v0: float | None) -> float | None:
    return v0 if v0 is not None else (v6 if v6 is not None else v12)


def mean_from_timepoints(vals: List[float | None]) -> float | None:
    clean = [v for v in vals if v is not None]
    if not clean:
        return None
    return float(np.mean(clean))


def build_manual_features(
    inputs: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """
    inputs[var] = {"t12": ..., "t6": ..., "t0": ...} (strings or for MechVent: option)
    Returns: (feats, errors, warnings)
    """
    errors: List[str] = []
    warnings: List[str] = []
    feats: Dict[str, Any] = {}

    # Clinician-facing specs (hard-ish ranges)
    # (You can tweak these anytime)
    specs = {
        "HR":       {"label": "Heart rate",         "units": "bpm",   "lo": 0,   "hi": 250},
        "RespRate": {"label": "Respiratory rate",   "units": "/min",  "lo": 0,   "hi": 80},
        "SysBP":    {"label": "Systolic BP",        "units": "mmHg",  "lo": 0,   "hi": 300},
        "DiasBP":   {"label": "Diastolic BP",       "units": "mmHg",  "lo": 0,   "hi": 200},
        "MAP":      {"label": "Mean arterial BP",   "units": "mmHg",  "lo": 0,   "hi": 200},
        "Temp":     {"label": "Temperature",        "units": "°C",    "lo": 25,  "hi": 45},
        "SaO2":     {"label": "Oxygen saturation",  "units": "%",     "lo": 0,   "hi": 100},
        "FiO2":     {"label": "FiO₂",               "units": "",      "lo": 0.21,"hi": 1.0},
        "pH":       {"label": "pH",                 "units": "",      "lo": 6.8, "hi": 7.8},
        "Lactate":  {"label": "Lactate",            "units": "mmol/L", "lo": 0,   "hi": 30},
        "Glucose":  {"label": "Glucose",            "units": "mmol/L", "lo": 0,   "hi": 60},
        # Add more if you want stricter validation for other labs later
    }

    # MechVent is special (Yes/No/Unknown)
    if "MechVent" in inputs:
        mv_vals: List[float | None] = []
        for tkey in ("t12", "t6", "t0"):
            opt = inputs["MechVent"].get(tkey, "Unknown")
            if opt == "Yes":
                mv_vals.append(1.0)
            elif opt == "No":
                mv_vals.append(0.0)
            else:
                mv_vals.append(None)

        feats["MechVent_was_measured"] = 1 if any(v is not None for v in mv_vals) else 0
        feats["MechVent_mean"] = mean_from_timepoints(mv_vals) if feats["MechVent_was_measured"] else np.nan
        last_mv = last_from_timepoints(mv_vals[0], mv_vals[1], mv_vals[2])
        feats["MechVent_last"] = last_mv if last_mv is not None else np.nan
        feats["MechVent_prop_on"] = feats["MechVent_mean"]  # same idea in this simplified manual mode

    # Other numeric vars
    for var, tvals in inputs.items():
        if var == "MechVent":
            continue

        v12_raw, e12 = parse_optional_float(tvals.get("t12", ""))
        v6_raw,  e6  = parse_optional_float(tvals.get("t6", ""))
        v0_raw,  e0  = parse_optional_float(tvals.get("t0", ""))

        # Friendly parse errors
        if e12:
            errors.append(f"{var} (12h): {e12}")
        if e6:
            errors.append(f"{var} (6h): {e6}")
        if e0:
            errors.append(f"{var} (now): {e0}")

        # Special: FiO2 accepts % or fraction; convert if needed, then validate fraction range
        if var == "FiO2":
            v12 = fio2_to_fraction(v12_raw) if v12_raw is not None else None
            v6  = fio2_to_fraction(v6_raw)  if v6_raw is not None else None
            v0  = fio2_to_fraction(v0_raw)  if v0_raw is not None else None

            # validate post-conversion
            spec = specs["FiO2"]
            for label, v in [("12h", v12), ("6h", v6), ("now", v0)]:
                if v is not None:
                    msg = validate_range(spec["label"], v, spec["lo"], spec["hi"], spec["units"])
                    if msg:
                        errors.append(f"{msg} (at {label}). Use fraction 0.21–1.0 or % 21–100.")
        else:
            v12, v6, v0 = v12_raw, v6_raw, v0_raw
            # validate if we have a spec
            if var in specs:
                spec = specs[var]
                for label, v in [("12h", v12), ("6h", v6), ("now", v0)]:
                    if v is not None:
                        msg = validate_range(spec["label"], v, spec["lo"], spec["hi"], spec["units"])
                        if msg:
                            errors.append(f"{msg} (at {label}).")

        any_measured = any(v is not None for v in (v12, v6, v0))
        feats[f"{var}_was_measured"] = 1 if any_measured else 0

        if not any_measured:
            feats[f"{var}_mean"] = np.nan
            feats[f"{var}_last"] = np.nan
        else:
            feats[f"{var}_mean"] = mean_from_timepoints([v12, v6, v0]) or np.nan
            last_v = last_from_timepoints(v12, v6, v0)
            feats[f"{var}_last"] = last_v if last_v is not None else np.nan

    return feats, errors, warnings


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.set_page_config(page_title="ICU Mortality Risk Demo", layout="centered")

st.title("ICU Mortality Risk Demo")
st.caption("Educational portfolio demo — not for clinical decision-making.")

with st.expander("What this tool is (and what it isn’t)"):
    st.write(
        """
        This demo shows how a risk tool could work end-to-end: it can take patient data, summarise it,
        and provide a **risk estimate** with **simple categories**.

        It is **not a validated clinical device** and the categories here are **demo thresholds**.
        Think of it like a **prototype** designed to demonstrate the workflow and UI, not a bedside tool.
        """
    )

# Load bundle
bundle_path = DEFAULT_BUNDLE_PATH
if not bundle_path.exists():
    st.error(
        "Model bundle not found. Expected: outputs/model_bundle.joblib\n\n"
        "Make sure it exists in the repo and is committed."
    )
    st.stop()

bundle = load_bundle(bundle_path)
model = bundle["model"]
imputer = bundle["imputer"]
feature_columns = bundle["feature_columns"]

thr_high = float(bundle.get("default_threshold", 0.5))
thr_medium = float(bundle.get("medium_threshold", DEFAULT_MEDIUM_THRESHOLD))

starter_vars = bundle.get("starter_vars", None)

# Keep clinician UI clean: show thresholds briefly, and hide the rest under “Technical details”
st.write("")
colA, colB = st.columns(2)
with colA:
    st.metric("High-risk threshold", f"{thr_high:.3f}")
with colB:
    st.metric("Medium threshold", f"{thr_medium:.3f}")

with st.expander("Technical details (for reviewers)"):
    st.write(f"- Model bundle: `{bundle_path.as_posix()}`")
    st.write(f"- Features expected: **{len(feature_columns)}**")
    if starter_vars is None:
        st.warning("Bundle does not store `starter_vars` (feature engineering drift risk).")
    else:
        st.info(f"Bundle starter_vars loaded: {len(starter_vars)} variables")

tabs = st.tabs(["Upload patient file (.txt)", "Manual entry"])

# ------------------------------------------------------------
# TAB 1: Upload .txt
# ------------------------------------------------------------
with tabs[0]:
    st.subheader("Upload patient file")
    st.write("Upload a single ICU patient `.txt` file (demo format) to generate a prediction.")

    uploaded = st.file_uploader("Patient file (.txt)", type=["txt"])

    if uploaded is not None:
        if uploaded.size > 2_000_000:
            st.error("File too large for this demo (max 2MB).")
            st.stop()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = Path(tmp.name)

        try:
            long_df = load_patient_long(tmp_path)
            feats = summarise_patient(long_df)

            X_imp, present_count, missing_features, X_aligned = align_and_impute(
                feats, feature_columns, imputer
            )

            prob = float(model.predict_proba(X_imp)[:, 1][0])
            band = risk_band(prob, thr_medium, thr_high)

            st.subheader("Result")
            st.metric("Predicted mortality risk", f"{prob*100:.2f}%")
            render_risk_pill(band)
            st.progress(min(max(prob, 0.0), 1.0))

            st.subheader("Data completeness")
            st.write(f"Features present: **{present_count}/{len(feature_columns)}**")
            st.write(f"Missing (imputed): **{len(missing_features)}**")
            if missing_features:
                with st.expander("Show missing features"):
                    st.write(missing_features)

            # Show ONLY non-null engineered values (so it’s not a wall of None)
            with st.expander("Show engineered values (non-missing only)"):
                row = X_aligned.iloc[0]
                non_missing = row[row.notna()]
                if non_missing.empty:
                    st.info("No engineered values were available.")
                else:
                    st.dataframe(non_missing.reset_index().rename(columns={"index": "feature", 0: "value"}))

        except Exception:
            st.error("Something went wrong while predicting. Please check the uploaded file format.")
            st.stop()
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


# ------------------------------------------------------------
# TAB 2: Manual entry (12h / 6h / now)
# ------------------------------------------------------------
with tabs[1]:
    st.subheader("Manual entry")
    st.write(
        "Enter a few commonly recorded values. Leave anything unknown blank. "
        "This will approximate the features used by the model."
    )

    # We can default to starter_vars if stored, otherwise a sensible shortlist.
    default_vars = [
        "HR", "RespRate", "SysBP", "DiasBP", "MAP", "Temp",
        "SaO2", "FiO2", "pH", "Lactate", "Glucose", "MechVent"
    ]
    vars_to_use = starter_vars if isinstance(starter_vars, list) and len(starter_vars) > 0 else default_vars

    # Ensure MechVent is included if you want it
    if "MechVent" not in vars_to_use:
        vars_to_use = list(vars_to_use) + ["MechVent"]

    # Clinician labels + units shown in UI (friendly)
    display = {
        "HR": ("Heart rate", "bpm", "e.g., 88"),
        "RespRate": ("Respiratory rate", "/min", "e.g., 18"),
        "SysBP": ("Systolic BP", "mmHg", "e.g., 118"),
        "DiasBP": ("Diastolic BP", "mmHg", "e.g., 68"),
        "MAP": ("Mean arterial pressure", "mmHg", "e.g., 82"),
        "Temp": ("Temperature", "°C", "e.g., 36.8"),
        "SaO2": ("Oxygen saturation", "%", "e.g., 94"),
        "FiO2": ("FiO₂", "fraction or %", "e.g., 0.40 or 40"),
        "pH": ("pH", "", "e.g., 7.36"),
        "Lactate": ("Lactate", "mmol/L", "e.g., 1.8"),
        "Glucose": ("Glucose", "mmol/L", "e.g., 6.2"),
        "MechVent": ("Invasive ventilation", "", ""),
    }

    manual_inputs: Dict[str, Dict[str, Any]] = {}

    with st.form("manual_form", clear_on_submit=False):
        st.write("**Timepoints**: 12 hours ago / 6 hours ago / now")

        header = st.columns([2.2, 1.2, 1.2, 1.2])
        header[0].markdown("**Measure**")
        header[1].markdown("**12h ago**")
        header[2].markdown("**6h ago**")
        header[3].markdown("**Now**")

        for var in vars_to_use:
            label, units, placeholder = display.get(var, (var, "", ""))
            row = st.columns([2.2, 1.2, 1.2, 1.2])

            row[0].markdown(f"**{label}**  \n<small>{units}</small>", unsafe_allow_html=True)

            if var == "MechVent":
                # Unknown/No/Yes prevents “random numbers”
                manual_inputs[var] = {
                    "t12": row[1].selectbox("", ["Unknown", "No", "Yes"], key=f"{var}_12"),
                    "t6":  row[2].selectbox("", ["Unknown", "No", "Yes"], key=f"{var}_6"),
                    "t0":  row[3].selectbox("", ["Unknown", "No", "Yes"], key=f"{var}_0"),
                }
            else:
                # text_input allows blank; we validate + coerce to float.
                help_txt = placeholder
                if var == "FiO2":
                    help_txt = "Enter fraction 0.21–1.0 OR % 21–100 (we auto-convert)."

                manual_inputs[var] = {
                    "t12": row[1].text_input("", value="", placeholder=placeholder, key=f"{var}_12"),
                    "t6":  row[2].text_input("", value="", placeholder=placeholder, key=f"{var}_6"),
                    "t0":  row[3].text_input("", value="", placeholder=placeholder, key=f"{var}_0"),
                }

                # small guidance under the now-column (keeps UI tidy)
                if var in ("FiO2",):
                    row[3].caption(help_txt)

        submitted = st.form_submit_button("Calculate risk")

    if submitted:
        raw_engineered, errs, warns = build_manual_features(manual_inputs)

        if errs:
            st.error("Please fix the following before calculating risk:")
            for e in errs[:12]:
                st.write(f"- {e}")
            if len(errs) > 12:
                st.write(f"- …and {len(errs) - 12} more.")
            st.stop()

        # Align + impute + predict
        try:
            X_imp, present_count, missing_features, X_aligned = align_and_impute(
                raw_engineered, feature_columns, imputer
            )

            prob = float(model.predict_proba(X_imp)[:, 1][0])
            band = risk_band(prob, thr_medium, thr_high)

            st.subheader("Result")
            st.metric("Predicted mortality risk", f"{prob*100:.2f}%")
            render_risk_pill(band)
            st.progress(min(max(prob, 0.0), 1.0))

            st.subheader("Data completeness")
            st.write(f"Features present: **{present_count}/{len(feature_columns)}**")
            st.write(f"Missing (imputed): **{len(missing_features)}**")
            if missing_features:
                with st.expander("Show missing features"):
                    st.write(missing_features)

            with st.expander("Show engineered values used (non-missing only)"):
                row = X_aligned.iloc[0]
                non_missing = row[row.notna()]
                if non_missing.empty:
                    st.info("No engineered values were available.")
                else:
                    st.dataframe(non_missing.reset_index().rename(columns={"index": "feature", 0: "value"}))

        except Exception:
            st.error(
                "Something went wrong while calculating risk. "
                "This usually means too many fields were left blank or a value format was unexpected."
            )
            st.stop()
