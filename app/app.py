"""
Streamlit demo app (educational).

Notes:
- Requires a saved model bundle at outputs/model_bundle.joblib
- Allows a user to upload a single patient file (.txt) OR manually enter observations
- Returns calibrated probability + risk band
- Includes a clinician-friendly “Why this estimate?” section (“pushes risk”)
"""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Path setup (robust imports)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # project root (one level above /app)

# Ensure project root is importable so `import scripts...` works everywhere
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step_01_load_raw import load_patient_long  # type: ignore
from scripts.step_02_batch_features import summarise_patient  # type: ignore


# -----------------------------
# Constants / defaults
# -----------------------------
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "outputs" / "model_bundle.joblib"
DEFAULT_MEDIUM_THRESHOLD = 0.10  # LOW < 0.10, MEDIUM [0.10..HIGH), HIGH >= HIGH_THR
MAX_UPLOAD_BYTES = 2_000_000

# “Strict mode” minimums (tune these if you want)
MIN_RAW_VARIABLES_FOR_STRICT_MANUAL = 6
MIN_PRESENT_FEATURES_FOR_STRICT_UPLOAD = 20


# -----------------------------
# Bundle loader
# -----------------------------
@st.cache_resource
def load_bundle(bundle_path: Path) -> dict:
    return joblib.load(bundle_path)


# -----------------------------
# Utility: feature name prettifier
# -----------------------------
def pretty_label(feature_name: str) -> str:
    """
    Convert model feature names to clinician-friendly labels.
    Examples:
      HeartRate_mean -> Heart rate (mean)
      FiO2_last -> FiO₂ (most recent)
      MechVent_prop_on -> Mechanical ventilation (proportion on)
      GCS_last -> GCS (most recent)
    """
    s = feature_name

    # common replacements
    s = s.replace("FiO2", "FiO₂")
    s = s.replace("SaO2", "SaO₂")
    s = s.replace("MechVent", "Mechanical ventilation")
    s = s.replace("HeartRate", "Heart rate")
    s = s.replace("RespRate", "Respiratory rate")
    s = s.replace("SysBP", "Systolic BP")
    s = s.replace("DiasBP", "Diastolic BP")
    s = s.replace("MeanBP", "Mean arterial pressure")
    s = s.replace("HCT", "Haematocrit")
    s = s.replace("WBC", "White cell count")
    s = s.replace("BUN", "Urea (BUN)")

    # suffixes
    if s.endswith("_mean"):
        s = s.replace("_mean", " (mean)")
    elif s.endswith("_last"):
        s = s.replace("_last", " (most recent)")
    elif s.endswith("_was_measured"):
        s = s.replace("_was_measured", " (was measured)")
    elif s.endswith("_prop_on"):
        s = s.replace("_prop_on", " (proportion on)")

    # spacing
    s = s.replace("_", " ")
    return s


# -----------------------------
# Align + impute safely (NO pd.NA reaching sklearn)
# -----------------------------
def align_and_impute(
    feats: dict,
    feature_columns: list[str],
    imputer: Any,
) -> tuple[pd.DataFrame, int, list[str], pd.DataFrame]:
    """
    Build a one-row dataframe in training feature order, report missing features,
    then return imputed X plus aligned (pre-imputation) X.

    IMPORTANT:
    - sklearn cannot handle pd.NA in this path reliably; we convert everything to np.nan
    """
    X = pd.DataFrame([feats])

    # Ensure all expected columns exist
    for c in feature_columns:
        if c not in X.columns:
            X[c] = np.nan

    # Keep ONLY training columns in correct order
    X = X[feature_columns]

    # Replace pd.NA / None with np.nan, then coerce to numeric
    X = X.replace({pd.NA: np.nan})
    X = X.where(pd.notna(X), np.nan)

    # to_numeric per column
    for col in X.columns:
        try:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        except Exception:
            X[col] = np.nan

    # Missing feature reporting (pre-imputation)
    missing_mask = X.isna().iloc[0]
    missing_features = X.columns[missing_mask].tolist()
    present_count = int((~missing_mask).sum())

    # Impute
    X_imp_arr = imputer.transform(X)
    X_imp = pd.DataFrame(X_imp_arr, columns=feature_columns)

    return X_imp, present_count, missing_features, X


# -----------------------------
# Explainability: “pushes risk”
# -----------------------------
def top_pushes_risk(
    X_imp: pd.DataFrame,
    feature_columns: list[str],
    model: Any,
    imputer: Any,
    k: int = 5,
) -> pd.DataFrame:
    """
    Simple, stable, no-SHAP explanation:
    - Uses RF feature_importances_ (global)
    - Uses imputer.statistics_ as “typical” baseline (median from training)
    - Computes a “push score” = importance * (value - typical)
    - Shows top positive pushes only

    This is not causation — it’s an intuitive “what’s unusual in the risky direction” display.
    """
    if not hasattr(model, "estimators_") and not hasattr(model, "base_estimator"):
        return pd.DataFrame(columns=["Factor", "Your value", "Typical", "Push score"])

    # Try to get importances from underlying RF (CalibratedClassifierCV wraps estimator)
    base = getattr(model, "estimator", None)
    if base is None and hasattr(model, "calibrated_classifiers_"):
        # older sklearn structures
        base = getattr(model.calibrated_classifiers_[0], "estimator", None)

    rf = base if base is not None else model

    if not hasattr(rf, "feature_importances_"):
        return pd.DataFrame(columns=["Factor", "Your value", "Typical", "Push score"])

    importances = np.asarray(rf.feature_importances_, dtype=float)
    if len(importances) != len(feature_columns):
        return pd.DataFrame(columns=["Factor", "Your value", "Typical", "Push score"])

    # Typical baseline (training medians) if available
    typical = getattr(imputer, "statistics_", None)
    if typical is None or len(typical) != len(feature_columns):
        typical = np.zeros(len(feature_columns), dtype=float)

    x = X_imp.iloc[0].to_numpy(dtype=float)
    typical = np.asarray(typical, dtype=float)

    push = importances * (x - typical)

    rows = []
    for i, fname in enumerate(feature_columns):
        rows.append(
            {
                "Factor": pretty_label(fname),
                "Your value": float(x[i]),
                "Typical": float(typical[i]),
                "Push score": float(push[i]),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("Push score", ascending=False)

    # Keep only positive pushes (increasing risk direction)
    df = df[df["Push score"] > 0].head(k).copy()
    df["Push score"] = df["Push score"].round(4)

    return df


# -----------------------------
# Risk band logic
# -----------------------------
def risk_band(prob: float, thr_low: float, thr_high: float) -> str:
    if prob < thr_low:
        return "LOW"
    if prob < thr_high:
        return "MEDIUM"
    return "HIGH"


def band_dot(band: str) -> str:
    return {"LOW": "🟢", "MEDIUM": "🟠", "HIGH": "🔴"}.get(band, "⚪")


# -----------------------------
# Manual input config
# -----------------------------
# Each manual row defines:
# - key base (used to build engineered vars)
# - label
# - widget type & constraints
MANUAL_FIELDS = [
    # Cardiac
    ("HeartRate", "Heart rate (bpm)", "num", 0.0, 250.0, 1.0),
    ("MeanBP", "Mean arterial pressure (NIBP) (mmHg)", "num", 0.0, 160.0, 1.0),
    ("SysBP", "Systolic BP (NIBP) (mmHg)", "num", 0.0, 300.0, 1.0),
    ("DiasBP", "Diastolic BP (NIBP) (mmHg)", "num", 0.0, 200.0, 1.0),
    # Respiratory
    ("RespRate", "Respiratory rate (breaths/min)", "num", 0.0, 80.0, 1.0),
    ("SaO2", "SaO₂ (%)", "num", 0.0, 100.0, 1.0),
    ("FiO2", "FiO₂ (fraction) — enter 0.21 to 1.00", "num", 0.21, 1.0, 0.01),
    ("MechVent", "Mechanical ventilation", "bool", None, None, None),
    # Blood gas
    ("pH", "pH", "num", 6.8, 7.8, 0.01),
    ("HCO3", "HCO₃⁻ (mmol/L)", "num", 0.0, 60.0, 0.1),
    # Bloods
    ("Na", "Sodium (Na) (mmol/L)", "num", 90.0, 200.0, 1.0),
    ("K", "Potassium (K) (mmol/L)", "num", 1.0, 9.0, 0.1),
    ("Mg", "Magnesium (Mg) (mmol/L)", "num", 0.2, 3.5, 0.1),
    ("Creatinine", "Creatinine (µmol/L)", "num", 0.0, 2000.0, 1.0),
    ("BUN", "Urea (BUN) (mg/dL)", "num", 0.0, 200.0, 0.5),
    ("Platelets", "Platelets (x10⁹/L)", "num", 0.0, 2000.0, 1.0),
    ("HCT", "Haematocrit (%)", "num", 0.0, 80.0, 0.5),
    ("Glucose", "Glucose (mmol/L)", "num", 0.0, 60.0, 0.1),
    ("Lactate", "Lactate (mmol/L)", "num", 0.0, 25.0, 0.1),
    # Other
    ("GCS", "GCS", "int", 3, 15, 1),
]

GROUPS = {
    "Cardiac": {"HeartRate", "MeanBP", "SysBP", "DiasBP"},
    "Respiratory": {"RespRate", "SaO2", "FiO2", "MechVent"},
    "Blood gas": {"pH", "HCO3"},
    "Bloods": {"Na", "K", "Mg", "Creatinine", "BUN", "Platelets", "HCT", "Glucose", "Lactate"},
    "Other": {"GCS"},
}

TIMEPOINTS = [("t12", "12h ago"), ("t6", "6h ago"), ("tnow", "Now")]


def render_manual_table() -> dict:
    """
    Render manual input grid.
    Returns raw input dict, structured as raw[var][timepoint] = value/None
    """
    raw: dict[str, dict[str, Any]] = {k: {} for (k, *_rest) in MANUAL_FIELDS}

    for gname, keys in GROUPS.items():
        st.subheader(gname)

        # header row
        cols = st.columns([2.2, 1.2, 1.2, 1.2])
        cols[0].markdown("**Measurement**")
        cols[1].markdown("**12h ago**")
        cols[2].markdown("**6h ago**")
        cols[3].markdown("**Now**")

        for (var, label, wtype, vmin, vmax, step) in MANUAL_FIELDS:
            if var not in keys:
                continue

            row = st.columns([2.2, 1.2, 1.2, 1.2])
            row[0].write(label)

            for j, (tkey, tlabel) in enumerate(TIMEPOINTS, start=1):
                widget_key = f"manual__{var}__{tkey}"

                if wtype == "bool":
                    raw_val = row[j].selectbox(
                        "",
                        options=["", "No", "Yes"],
                        index=0,
                        key=widget_key,
                        help="Leave blank if unknown.",
                    )
                    if raw_val == "":
                        raw[var][tkey] = None
                    else:
                        raw[var][tkey] = 1.0 if raw_val == "Yes" else 0.0

                elif wtype == "int":
                    # allow blank via text_input -> parse later
                    txt = row[j].text_input(
                        "",
                        value="",
                        key=widget_key,
                        placeholder="",
                        help=f"Enter {int(vmin)}–{int(vmax)} or leave blank.",
                    )
                    raw[var][tkey] = txt

                else:
                    # numeric entry but allow blank using text_input so we don’t force 0s
                    txt = row[j].text_input(
                        "",
                        value="",
                        key=widget_key,
                        placeholder="",
                        help=f"Enter {vmin}–{vmax} or leave blank.",
                    )
                    raw[var][tkey] = txt

        st.divider()

    return raw


def parse_manual_raw(raw: dict) -> tuple[dict, list[str], int]:
    """
    Parse raw manual strings into numbers where possible.
    Returns:
      - cleaned raw values (float or None)
      - human-readable warnings
      - count of raw variables entered (non-missing values across timepoints)
    """
    warnings: list[str] = []
    entered = 0

    # build quick lookup for constraints
    constraints = {var: (wtype, vmin, vmax) for (var, _lab, wtype, vmin, vmax, _step) in MANUAL_FIELDS}

    cleaned: dict[str, dict[str, float | None]] = {k: {} for k in raw.keys()}

    for var, tvals in raw.items():
        wtype, vmin, vmax = constraints[var]

        for tkey, val in tvals.items():
            if val is None:
                cleaned[var][tkey] = None
                continue

            # already converted bool -> float
            if isinstance(val, (float, int)) and wtype == "bool":
                cleaned[var][tkey] = float(val)
                entered += 1
                continue

            # parse strings for int/num
            if isinstance(val, str):
                s = val.strip()
                if s == "":
                    cleaned[var][tkey] = None
                    continue

                try:
                    f = float(s)
                except Exception:
                    # clinician-friendly warning
                    warnings.append(f"“{constraints_label(var)}” at {timepoint_label(tkey)} wasn’t recognised as a number, so it was treated as unknown.")
                    cleaned[var][tkey] = None
                    continue

                # range check
                if vmin is not None and f < float(vmin):
                    warnings.append(f"“{constraints_label(var)}” at {timepoint_label(tkey)} must be ≥ {vmin}. It was treated as unknown.")
                    cleaned[var][tkey] = None
                    continue
                if vmax is not None and f > float(vmax):
                    warnings.append(f"“{constraints_label(var)}” at {timepoint_label(tkey)} must be ≤ {vmax}. It was treated as unknown.")
                    cleaned[var][tkey] = None
                    continue

                # int only if requested
                if wtype == "int":
                    f = float(int(round(f)))

                cleaned[var][tkey] = float(f)
                entered += 1
                continue

            # fallback
            cleaned[var][tkey] = None

    return cleaned, warnings, entered


def constraints_label(var: str) -> str:
    for (k, label, *_rest) in MANUAL_FIELDS:
        if k == var:
            return label
    return var


def timepoint_label(tkey: str) -> str:
    return {"t12": "12h ago", "t6": "6h ago", "tnow": "Now"}.get(tkey, tkey)


def engineer_from_manual(cleaned: dict) -> dict:
    """
    Turn manual timepoint values into engineered features the model expects.
    Rules:
      - If values exist: mean = mean of available timepoints, last = most recent available (Now > 6h > 12h)
      - If none exist: leave mean/last as np.nan (NOT invented)
      - MechVent additionally provides prop_on
      - Adds *_was_measured flags
    """
    feat: dict[str, Any] = {}

    # Most recent ordering
    order = ["t12", "t6", "tnow"]

    for (var, _label, _wtype, *_rest) in MANUAL_FIELDS:
        vals = [cleaned[var].get(t) for t in order]
        vals_num = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]

        if len(vals_num) == 0:
            feat[f"{var}_was_measured"] = 0
            feat[f"{var}_mean"] = np.nan
            feat[f"{var}_last"] = np.nan
            if var == "MechVent":
                feat["MechVent_prop_on"] = np.nan
            continue

        feat[f"{var}_was_measured"] = 1
        feat[f"{var}_mean"] = float(np.mean(vals_num))

        # last = most recent available among now -> 6h -> 12h
        last_val = None
        for t in ["tnow", "t6", "t12"]:
            v = cleaned[var].get(t)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                last_val = v
                break
        feat[f"{var}_last"] = float(last_val) if last_val is not None else np.nan

        if var == "MechVent":
            mv = np.array(vals_num, dtype=float)
            feat["MechVent_prop_on"] = float(np.mean(mv)) if mv.size else np.nan

    return feat


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="ICU Mortality Risk Demo", layout="centered")

st.title("ICU Mortality Risk Demo")
st.caption("For demonstration and education only — not a clinical decision tool.")

# Bundle is fixed (repo-local)
bundle_path = DEFAULT_BUNDLE_PATH
if not bundle_path.exists():
    st.error(
        "Model bundle not found in the expected location. "
        "Make sure outputs/model_bundle.joblib is committed in the repo."
    )
    st.stop()

bundle = load_bundle(bundle_path)
model = bundle["model"]
imputer = bundle["imputer"]
feature_columns = bundle["feature_columns"]

thr_high = float(bundle.get("default_threshold", 0.5))
thr_low = float(bundle.get("medium_threshold", DEFAULT_MEDIUM_THRESHOLD))  # low/high boundary naming legacy
starter_vars = bundle.get("starter_vars", None)

# Friendly wording: LOW threshold is thr_low
low_thr = thr_low
high_thr = thr_high

with st.expander("What this tool does", expanded=False):
    st.write(
        "This tool estimates ICU mortality risk from observations (vitals / labs / blood gases). "
        "It can either read a patient `.txt` file (demo format) or you can enter observations manually."
    )
    st.write(
        "If some observations are missing, you can choose:"
    )
    st.markdown(
        "- **Strict mode:** no estimate unless enough real data is provided.\n"
        "- **Estimate mode:** missing values are filled using typical values from the model’s training data "
        "(this will be clearly disclosed)."
    )

with st.expander("What data can I upload?", expanded=False):
    st.write("Upload a single patient `.txt` file in the same format used in this demo project (e.g., PhysioNet-style).")
    st.write("If you’re not sure, use **Manual entry** instead.")

# Keep selection sticky
if "input_method" not in st.session_state:
    st.session_state["input_method"] = "Manual entry"

st.subheader("Choose input method")
input_method = st.radio(
    "",
    ["Upload patient file", "Manual entry"],
    index=1 if st.session_state["input_method"] == "Manual entry" else 0,
    key="input_method_radio",
)
st.session_state["input_method"] = input_method

# Prediction behaviour (sticky)
if "mode" not in st.session_state:
    st.session_state["mode"] = "Strict mode (needs enough real data)"

mode = st.selectbox(
    "Prediction behaviour",
    ["Strict mode (needs enough real data)", "Estimate mode (fills missing values)"],
    index=0 if st.session_state["mode"].startswith("Strict") else 1,
    help="Strict mode refuses to estimate if too much data is missing. Estimate mode will predict but will clearly show what was filled in.",
)
st.session_state["mode"] = mode


def show_result_block(
    prob: float,
    low_thr: float,
    high_thr: float,
    X_imp: pd.DataFrame,
    missing_features: list[str],
    present_count: int,
    raw_entered_count: int | None = None,
    aligned_X: pd.DataFrame | None = None,
):
    band = risk_band(prob, low_thr, high_thr)

    st.subheader("Result")
    st.write(f"{band_dot(band)} **{band} RISK**")
    st.metric("Estimated mortality risk", f"{prob*100:.2f}%")
    st.caption(f"Model probability: {prob:.6f}")
    st.caption(f"Risk bands: LOW < {low_thr:.2f}, MEDIUM {low_thr:.2f}–{high_thr:.3f}, HIGH ≥ {high_thr:.3f}")

    st.subheader("What data was missing (and filled in by the model)")
    if len(missing_features) == 0:
        st.write("None — all model features were provided.")
    else:
        # Present missing in clinician-friendly grouped way
        groups_hit = set()
        for mf in missing_features:
            base = mf.split("_")[0]
            for gname, keys in GROUPS.items():
                if base in keys:
                    groups_hit.add(gname)
        if groups_hit:
            st.write(", ".join(sorted(groups_hit)))
        else:
            # fallback: show readable names
            st.write(", ".join(pretty_label(m) for m in missing_features[:30]))

    st.subheader("Why this risk estimate?")
    st.caption("These are the top factors that increased the model’s estimate. This is not proof of causation.")
    pushes = top_pushes_risk(X_imp, feature_columns, model, imputer, k=5)
    if pushes.empty:
        st.write("No clear top drivers were available for display.")
    else:
        # show only factor + your value (hide push score unless technical section)
        disp = pushes[["Factor", "Your value"]].copy()
        st.dataframe(disp, use_container_width=True, hide_index=True)

    # Export (simple)
    st.subheader("Export")
    export = {
        "probability": prob,
        "risk_band": band,
        "threshold_low": low_thr,
        "threshold_high": high_thr,
        "present_features": present_count,
        "missing_features": [pretty_label(m) for m in missing_features],
    }
    if raw_entered_count is not None:
        export["raw_variables_entered"] = raw_entered_count

    st.download_button(
        "Download summary (JSON)",
        data=json.dumps(export, indent=2).encode("utf-8"),
        file_name="icu_mortality_demo_summary.json",
        mime="application/json",
    )

    # Technical details (single expander; NO nested expanders)
    with st.expander("Technical details", expanded=False):
        tab1, tab2, tab3 = st.tabs(["Engineered values", "Model feature alignment", "Environment"])
        with tab1:
            if aligned_X is not None:
                # show only non-null engineered values
                nonnull = aligned_X.iloc[0].dropna()
                if nonnull.empty:
                    st.write("No engineered values available.")
                else:
                    dfv = nonnull.reset_index()
                    dfv.columns = ["Feature", "Value"]
                    dfv["Feature"] = dfv["Feature"].apply(pretty_label)
                    st.dataframe(dfv, use_container_width=True, hide_index=True)
            else:
                st.write("No engineered-value table available in this path.")

        with tab2:
            st.write(f"Features present: {present_count}/{len(feature_columns)}")
            st.write(f"Missing features (imputed): {len(missing_features)}")
            if len(missing_features) > 0:
                st.write("Missing:")
                st.write("\n".join(f"- {pretty_label(m)}" for m in missing_features))

        with tab3:
            import sklearn
            st.write(f"Python: {sys.version.split()[0]}")
            st.write(f"scikit-learn: {sklearn.__version__}")
            st.write(f"numpy: {np.__version__}")
            st.write(f"pandas: {pd.__version__}")
            st.write(f"joblib: {joblib.__version__}")


# -----------------------------
# Upload mode
# -----------------------------
if input_method == "Upload patient file":
    st.subheader("Upload patient file")

    uploaded = st.file_uploader("Upload patient file (.txt)", type=["txt"])
    if uploaded is None:
        st.stop()

    if uploaded.size > MAX_UPLOAD_BYTES:
        st.error("File too large for this demo (max 2MB).")
        st.stop()

    # Use temp file (no repo writes)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = Path(tmp.name)

    try:
        long_df = load_patient_long(tmp_path)
        feats = summarise_patient(long_df)

        X_imp, present_count, missing_features, X_aligned = align_and_impute(feats, feature_columns, imputer)

        # strict mode gate
        if mode.startswith("Strict"):
            if present_count < MIN_PRESENT_FEATURES_FOR_STRICT_UPLOAD:
                st.warning(
                    "Insufficient data to produce a risk estimate in **Strict mode**.\n\n"
                    f"Provided model features: {present_count}/{len(feature_columns)}.\n\n"
                    "Switch to **Estimate mode** if you want an approximate estimate that fills missing values "
                    "using typical training values."
                )
                st.stop()

        # estimate mode disclosure
        if mode.startswith("Estimate"):
            st.info(
                f"Estimate mode is ON: {len(missing_features)} model features were missing and were filled using typical values from training data."
            )

        prob = float(model.predict_proba(X_imp)[:, 1][0])
        show_result_block(prob, low_thr, high_thr, X_imp, missing_features, present_count, aligned_X=X_aligned)

    except Exception as e:
        st.error("Something went wrong while predicting.")
        st.exception(e)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


# -----------------------------
# Manual mode
# -----------------------------
else:
    st.subheader("Manual entry")
    st.caption("Enter observations at up to three timepoints. Leave blank if unknown.")

    raw = render_manual_table()

    cleaned, warnings, entered_count = parse_manual_raw(raw)

    st.subheader("Data check")
    st.write(f"Variables entered: **{entered_count}**")

    for w in warnings[:8]:
        st.warning(w)
    if len(warnings) > 8:
        st.warning(f"...and {len(warnings) - 8} more entries were treated as unknown.")

    # Build engineered features from manual
    engineered = engineer_from_manual(cleaned)

    # Calculate button (keep app on manual selection)
    clicked = st.button("Calculate risk", type="primary")
    if not clicked:
        st.stop()

    try:
        X_imp, present_count, missing_features, X_aligned = align_and_impute(engineered, feature_columns, imputer)

        # strict mode gate (manual uses raw-entered variables)
        if mode.startswith("Strict"):
            if entered_count < MIN_RAW_VARIABLES_FOR_STRICT_MANUAL:
                st.warning(
                    "Insufficient data to produce a risk estimate in **Strict mode**.\n\n"
                    f"You entered {entered_count} values. Minimum recommended: {MIN_RAW_VARIABLES_FOR_STRICT_MANUAL}.\n\n"
                    "Switch to **Estimate mode** if you want an approximate estimate that fills missing values "
                    "using typical training values."
                )
                st.stop()

        # estimate mode disclosure
        if mode.startswith("Estimate"):
            st.info(
                f"Estimate mode is ON: {len(missing_features)} model features were missing and were filled using typical values from training data."
            )

        prob = float(model.predict_proba(X_imp)[:, 1][0])

        show_result_block(
            prob,
            low_thr,
            high_thr,
            X_imp,
            missing_features,
            present_count,
            raw_entered_count=entered_count,
            aligned_X=X_aligned,
        )

    except Exception as e:
        st.error("Something went wrong while predicting.")
        st.exception(e)
