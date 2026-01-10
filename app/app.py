"""
Streamlit demo app (educational).

Main module for Streamlit Cloud:
  app/app.py

Notes:
- Requires a saved model bundle at outputs/model_bundle.joblib
- Upload mode: user uploads one patient .txt -> parse -> engineer -> predict
- Manual mode: user enters a subset of observations at up to three timepoints (12h/6h/now)
  -> we engineer mean/latest + was_measured flags for the variables we have.
- Strict vs Estimate:
    Strict (default): only produces a result if "enough" core observations are entered.
    Estimate: always produces a result (more likely to rely on training-default fills).
"""

from __future__ import annotations

import sys
from pathlib import Path
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
DEFAULT_MEDIUM_THRESHOLD = 0.10  # LOW < 0.10, MEDIUM [0.10..HIGH), HIGH >= default_threshold


# -----------------------------
# Clinician-facing measurement specs
# -----------------------------
@dataclass(frozen=True)
class MeasureSpec:
    key: str                      # base variable name used in training feature engineering
    label: str                    # clinician label (with units)
    section: str                  # grouping
    kind: str                     # "num" or "bool"
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    help_text: str = ""


SECTIONS_ORDER = ["Cardiac", "Respiratory", "Blood gas", "Bloods", "Other"]
TIMEPOINTS = [("t12", "12h ago"), ("t6", "6h ago"), ("tnow", "Now")]

MEASURES: List[MeasureSpec] = [
    # Cardiac
    MeasureSpec("HR", "Heart rate (bpm)", "Cardiac", "num", 0, 250,
                "Enter a number between 0 and 250. Leave blank if unknown."),
    MeasureSpec("NIMAP", "Mean arterial pressure (NIBP) (mmHg)", "Cardiac", "num", 0, 200,
                "Enter a number between 0 and 200. Leave blank if unknown."),
    MeasureSpec("NISysABP", "Systolic BP (NIBP) (mmHg)", "Cardiac", "num", 0, 300,
                "Enter a number between 0 and 300. Leave blank if unknown."),
    MeasureSpec("NIDiasABP", "Diastolic BP (NIBP) (mmHg)", "Cardiac", "num", 0, 200,
                "Enter a number between 0 and 200. Leave blank if unknown."),

    # Respiratory
    MeasureSpec("RespRate", "Respiratory rate (breaths/min)", "Respiratory", "num", 0, 80,
                "Enter a number between 0 and 80. Leave blank if unknown."),
    MeasureSpec("SaO2", "SaO₂ (%)", "Respiratory", "num", 0, 100,
                "Enter a number between 0 and 100. Leave blank if unknown."),
    MeasureSpec("FiO2", "FiO₂ (fraction) — enter 0.21 to 1.00", "Respiratory", "num", 0.21, 1.00,
                "Enter a fraction between 0.21 and 1.00 (e.g., 0.28). Leave blank if unknown."),
    MeasureSpec("MechVent", "Mechanical ventilation", "Respiratory", "bool",
                help_text="Choose Yes/No. Leave blank if unknown."),

    # Blood gas
    MeasureSpec("pH", "pH", "Blood gas", "num", 6.8, 7.8,
                "Enter a value between 6.8 and 7.8. Leave blank if unknown."),
    MeasureSpec("HCO3", "HCO₃⁻ (mmol/L)", "Blood gas", "num", 0, 60,
                "Enter a value between 0 and 60. Leave blank if unknown."),

    # Bloods
    MeasureSpec("Na", "Sodium (Na) (mmol/L)", "Bloods", "num", 80, 200,
                "Enter a value between 80 and 200. Leave blank if unknown."),
    MeasureSpec("K", "Potassium (K) (mmol/L)", "Bloods", "num", 1, 10,
                "Enter a value between 1 and 10. Leave blank if unknown."),
    MeasureSpec("Mg", "Magnesium (Mg) (mmol/L)", "Bloods", "num", 0, 5,
                "Enter a value between 0 and 5. Leave blank if unknown."),
    MeasureSpec("Creatinine", "Creatinine (µmol/L)", "Bloods", "num", 0, 3000,
                "Enter a value between 0 and 3000. Leave blank if unknown."),
    MeasureSpec("Urea", "Urea (BUN) (mg/dL)", "Bloods", "num", 0, 200,
                "Enter a value between 0 and 200. Leave blank if unknown."),
    MeasureSpec("Platelets", "Platelets (x10⁹/L)", "Bloods", "num", 0, 2000,
                "Enter a value between 0 and 2000. Leave blank if unknown."),
    MeasureSpec("HCT", "Haematocrit (%)", "Bloods", "num", 0, 80,
                "Enter a value between 0 and 80. Leave blank if unknown."),
    MeasureSpec("Glucose", "Glucose (mmol/L)", "Bloods", "num", 0, 60,
                "Enter a value between 0 and 60. Leave blank if unknown."),
    MeasureSpec("Lactate", "Lactate (mmol/L)", "Bloods", "num", 0, 30,
                "Enter a value between 0 and 30. Leave blank if unknown."),

    # Other
    MeasureSpec("GCS", "GCS", "Other", "num", 3, 15,
                "Enter a value between 3 and 15. Leave blank if unknown."),
]


MEASURE_BY_KEY = {m.key: m for m in MEASURES}


# -----------------------------
# Bundle loading + utilities
# -----------------------------
@st.cache_resource
def load_bundle(bundle_path: Path) -> dict:
    return joblib.load(bundle_path)


def safe_float(x: Any) -> float:
    # robust conversion for thresholds etc.
    if x is None:
        return float("nan")
    try:
        if pd.isna(x):
            return float("nan")
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return float("nan")


def get_base_model_for_shap(model: Any) -> Optional[Any]:
    """
    Try to recover an uncalibrated estimator suitable for SHAP explanations.
    CalibratedClassifierCV wraps an underlying estimator.
    """
    # If it's already a tree model (RF), return it
    if model is None:
        return None

    # CalibratedClassifierCV often stores base estimator differently across versions
    if hasattr(model, "estimator"):
        est = getattr(model, "estimator")
        if est is not None:
            return est

    if hasattr(model, "calibrated_classifiers_"):
        ccs = getattr(model, "calibrated_classifiers_")
        if ccs and hasattr(ccs[0], "estimator"):
            return ccs[0].estimator

    return None


def pretty_feature_name(feat: str) -> str:
    """
    Convert model feature names like 'RespRate_mean' into clinician-friendly text.
    """
    if feat.endswith("_mean"):
        base = feat[:-5]
        base_label = MEASURE_BY_KEY.get(base, None)
        return f"{base_label.label if base_label else base} — average"
    if feat.endswith("_last"):
        base = feat[:-5]
        base_label = MEASURE_BY_KEY.get(base, None)
        return f"{base_label.label if base_label else base} — latest"
    if feat.endswith("_was_measured"):
        base = feat[:-13]
        base_label = MEASURE_BY_KEY.get(base, None)
        return f"{base_label.label if base_label else base} — recorded?"
    if feat == "MechVent_prop_on":
        return "Mechanical ventilation — proportion on"
    # fallback
    return feat.replace("_", " ").strip()


# -----------------------------
# Manual entry parsing + engineering
# -----------------------------
def parse_numeric(text: str, spec: MeasureSpec) -> Tuple[Optional[float], Optional[str]]:
    """
    Returns (value_or_none, user_facing_error_or_none).
    Empty -> (None, None)
    Invalid/out-of-range -> (None, error message)
    """
    s = (text or "").strip()
    if s == "":
        return None, None

    try:
        val = float(s)
    except Exception:
        return None, f"{spec.label}: please enter a number."

    if spec.min_val is not None and val < spec.min_val:
        return None, f"{spec.label}: please enter a value between {spec.min_val:g} and {spec.max_val:g}."
    if spec.max_val is not None and val > spec.max_val:
        return None, f"{spec.label}: please enter a value between {spec.min_val:g} and {spec.max_val:g}."

    return val, None


def parse_bool(choice: str, spec: MeasureSpec) -> Tuple[Optional[int], Optional[str]]:
    """
    Returns (0/1/None, error).
    """
    c = (choice or "").strip()
    if c == "":
        return None, None
    if c.lower() == "yes":
        return 1, None
    if c.lower() == "no":
        return 0, None
    return None, f"{spec.label}: please choose Yes or No."


def engineer_from_timepoints(
    tp_values: Dict[str, Dict[str, Any]],
    starter_vars: List[str],
) -> Tuple[Dict[str, Any], int, List[str]]:
    """
    tp_values: { base_var: {t12: raw, t6: raw, tnow: raw} } with raw already parsed to float/int/None
    starter_vars: list of base variables used during training feature engineering
    Returns:
      engineered_feats (dict),
      core_entered_count (count of base vars with any entered value),
      core_entered_keys (list of base vars entered)
    """
    feats: Dict[str, Any] = {}
    entered_vars: List[str] = []

    for var in starter_vars:
        spec = MEASURE_BY_KEY.get(var, None)

        vals = []
        last_val = None

        if var in tp_values:
            # preserve time order: 12h -> 6h -> now
            ordered = [tp_values[var].get("t12"), tp_values[var].get("t6"), tp_values[var].get("tnow")]
            for v in ordered:
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(v)
                    last_val = v

        if len(vals) == 0:
            feats[f"{var}_was_measured"] = 0
            feats[f"{var}_mean"] = np.nan
            feats[f"{var}_last"] = np.nan
            continue

        entered_vars.append(var)
        feats[f"{var}_was_measured"] = 1

        # numeric mean/last
        try:
            feats[f"{var}_mean"] = float(np.mean(vals))
        except Exception:
            feats[f"{var}_mean"] = np.nan

        feats[f"{var}_last"] = float(last_val) if last_val is not None else np.nan

        # special handling for MechVent
        if var.lower() == "mechvent":
            mv = np.array(vals, dtype=float)
            feats["MechVent_prop_on"] = float(np.mean(mv)) if mv.size > 0 else np.nan
            feats["MechVent_last"] = float(last_val) if last_val is not None else np.nan

    return feats, len(entered_vars), entered_vars


# -----------------------------
# Alignment + imputation (critical: no pd.NA enters sklearn)
# -----------------------------
def align_and_impute(
    feats: Dict[str, Any],
    feature_columns: List[str],
    imputer: Any,
) -> Tuple[pd.DataFrame, int, List[str], pd.DataFrame]:
    """
    Returns:
      X_imp (1-row DataFrame),
      present_count (non-missing in aligned row, pre-imputation),
      missing_feature_names (model feature column names missing),
      X_aligned (1-row DataFrame pre-imputation, aligned)
    """
    X = pd.DataFrame([feats])

    for c in feature_columns:
        if c not in X.columns:
            X[c] = np.nan

    X = X[feature_columns]

    # ensure pure numeric + np.nan (NOT pd.NA)
    X = X.where(pd.notna(X), np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")

    missing_mask = X.isna().iloc[0]
    missing_features = X.columns[missing_mask].tolist()
    present_count = int((~missing_mask).sum())

    X_imp = pd.DataFrame(imputer.transform(X), columns=feature_columns)

    return X_imp, present_count, missing_features, X


# -----------------------------
# Risk bands + reliability
# -----------------------------
def risk_band(prob: float, thr_medium: float, thr_high: float) -> str:
    if prob < thr_medium:
        return "LOW"
    if prob < thr_high:
        return "MEDIUM"
    return "HIGH"


def reliability_badge(core_entered: int, core_total: int, imputed_count: int, model_total: int) -> Tuple[str, str]:
    """
    Returns (badge_label, badge_emoji)
    Simple, clinician-friendly heuristic:
      - High: core >= 80% and imputed <= 20%
      - Moderate: core >= 50% and imputed <= 40%
      - Low: otherwise
    """
    core_pct = (core_entered / max(core_total, 1)) * 100.0
    imp_pct = (imputed_count / max(model_total, 1)) * 100.0

    if core_pct >= 80 and imp_pct <= 20:
        return "High reliability", "🟢"
    if core_pct >= 50 and imp_pct <= 40:
        return "Moderate reliability", "🟠"
    return "Low reliability", "🔴"


def strict_minimum_ok(core_by_section: Dict[str, int]) -> Tuple[bool, str]:
    """
    Strict mode rule:
      - At least 6 observations total across core variables
      - At least 1 from Cardiac/Respiratory (vitals)
      - At least 1 from Blood gas/Bloods (labs)
    """
    total = sum(core_by_section.values())
    vitals = core_by_section.get("Cardiac", 0) + core_by_section.get("Respiratory", 0)
    labs = core_by_section.get("Blood gas", 0) + core_by_section.get("Bloods", 0)

    if total < 6:
        return False, "Please enter at least 6 observations in total."
    if vitals < 1:
        return False, "Please enter at least one vital sign (Cardiac or Respiratory)."
    if labs < 1:
        return False, "Please enter at least one lab value (Blood gas or Bloods)."
    return True, ""


# -----------------------------
# Explainable AI (top pushes risk)
# -----------------------------
def top_push_drivers(
    model_for_shap: Any,
    X_imp: pd.DataFrame,
    feature_columns: List[str],
    k: int = 5,
) -> pd.DataFrame:
    """
    Returns a table of top features that pushed risk UP for this sample.
    If SHAP fails, returns empty df.
    """
    try:
        import shap  # local import to avoid issues if removed later

        explainer = shap.TreeExplainer(model_for_shap)
        sv = explainer.shap_values(X_imp)

        # handle different SHAP return types
        if isinstance(sv, list):
            # binary: [class0, class1]
            shap_vals = np.array(sv[1])[0]
        else:
            arr = np.array(sv)
            if arr.ndim == 3:
                shap_vals = arr[0, :, 1]
            else:
                shap_vals = arr[0]

        shap_vals = shap_vals.astype(float)
        xrow = X_imp.iloc[0].to_numpy(dtype=float)

        rows = []
        for i, f in enumerate(feature_columns):
            rows.append((f, float(xrow[i]), float(shap_vals[i])))

        df = pd.DataFrame(rows, columns=["feature", "value", "shap"])

        # keep only positive pushes
        df_pos = df[df["shap"] > 0].copy()
        if df_pos.empty:
            # fallback: show most influential by absolute impact (still clinician-safe),
            # but label in the UI as "no strong increasing drivers".
            df_abs = df.copy()
            df_abs["abs_shap"] = df_abs["shap"].abs()
            df_abs = df_abs.sort_values("abs_shap", ascending=False).head(k)
            df_abs["feature"] = df_abs["feature"].apply(pretty_feature_name)
            df_abs = df_abs.rename(columns={"shap": "impact"})
            return df_abs[["feature", "value", "impact"]]

        df_pos = df_pos.sort_values("shap", ascending=False).head(k)
        df_pos["feature"] = df_pos["feature"].apply(pretty_feature_name)
        df_pos = df_pos.rename(columns={"shap": "impact"})
        return df_pos[["feature", "value", "impact"]]

    except Exception:
        return pd.DataFrame(columns=["feature", "value", "impact"])


# -----------------------------
# UI helpers
# -----------------------------
def section_counts(entered_vars: List[str]) -> Dict[str, int]:
    counts = {s: 0 for s in SECTIONS_ORDER}
    for v in entered_vars:
        spec = MEASURE_BY_KEY.get(v)
        if spec:
            counts[spec.section] = counts.get(spec.section, 0) + 1
    return counts


def build_missing_display(missing_features: List[str]) -> List[str]:
    # Filter out "recorded?" flags because clinicians don't care about those as missing
    cleaned = []
    for mf in missing_features:
        if mf.endswith("_was_measured"):
            continue
        cleaned.append(pretty_feature_name(mf))
    return sorted(cleaned)


def render_manual_entry_form() -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    Renders manual entry form and returns:
      tp_raw: {var: {t12,t6,tnow: raw string/choice}}
      errors: list of parse errors (empty until parsed after submit)
    """
    tp_raw: Dict[str, Dict[str, Any]] = {m.key: {} for m in MEASURES}
    errors: List[str] = []

    st.caption("Enter observations at up to three timepoints. Leave blank if unknown.")

    with st.form("manual_form", clear_on_submit=False):
        for section in SECTIONS_ORDER:
            items = [m for m in MEASURES if m.section == section]
            if not items:
                continue

            st.subheader(section)

            # header row
            hcols = st.columns([2.6, 1.2, 1.2, 1.2])
            hcols[0].markdown("**Measurement**")
            hcols[1].markdown("**12h ago**")
            hcols[2].markdown("**6h ago**")
            hcols[3].markdown("**Now**")

            for m in items:
                cols = st.columns([2.6, 1.2, 1.2, 1.2])
                cols[0].markdown(f"{m.label}")

                for tp_key, tp_label, col in zip(
                    [t[0] for t in TIMEPOINTS],
                    [t[1] for t in TIMEPOINTS],
                    cols[1:],
                ):
                    widget_key = f"{m.key}_{tp_key}"

                    if m.kind == "bool":
                        choice = col.selectbox(
                            label="",
                            options=["", "No", "Yes"],
                            index=0,
                            key=widget_key,
                            help=m.help_text if tp_key == "tnow" else None,
                        )
                        tp_raw[m.key][tp_key] = choice
                    else:
                        txt = col.text_input(
                            label="",
                            value=st.session_state.get(widget_key, ""),
                            key=widget_key,
                            help=m.help_text if tp_key == "tnow" else None,
                        )
                        tp_raw[m.key][tp_key] = txt

            st.divider()

        submitted = st.form_submit_button("Calculate risk")

    return tp_raw, errors


# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="ICU Mortality Risk Demo", layout="centered")

st.title("ICU Mortality Risk Demo")
st.caption("For demonstration and education only — not a clinical decision tool.")

with st.expander("What this tool does"):
    st.write(
        """
        This tool estimates ICU mortality risk from observations (vitals and labs) using a model trained on a public dataset.
        It is a **portfolio demonstration** to show an end-to-end workflow (data parsing → feature engineering → prediction).
        """
    )

with st.expander("What data can I upload?"):
    st.write(
        """
        Upload a single patient `.txt` file in the same format used by the training dataset.
        Alternatively, use Manual entry to type a small set of observations (leave blanks if unknown).
        """
    )


# -----------------------------
# Load bundle
# -----------------------------
bundle_path = DEFAULT_BUNDLE_PATH
if not bundle_path.exists():
    st.error(f"Model bundle not found at: {bundle_path.as_posix()}")
    st.stop()

bundle = load_bundle(bundle_path)
model = bundle["model"]
imputer = bundle["imputer"]
feature_columns: List[str] = list(bundle["feature_columns"])

thr_high = safe_float(bundle.get("default_threshold", 0.5))
if np.isnan(thr_high):
    thr_high = 0.5

thr_medium = safe_float(bundle.get("medium_threshold", DEFAULT_MEDIUM_THRESHOLD))
if np.isnan(thr_medium):
    thr_medium = DEFAULT_MEDIUM_THRESHOLD

starter_vars = bundle.get("starter_vars")
if starter_vars is None or not isinstance(starter_vars, list) or len(starter_vars) == 0:
    # fallback to the measures we expose
    starter_vars = [m.key for m in MEASURES]


# -----------------------------
# Input method
# -----------------------------
st.subheader("Choose input method")
input_method = st.radio(
    "",
    ["Upload patient file", "Manual entry"],
    key="input_method",
    horizontal=True,
)

st.divider()

# Mode selection (default strict)
st.subheader("Prediction mode")
mode = st.radio(
    "",
    ["Strict (recommended)", "Estimate (always runs)"],
    index=0,
    horizontal=True,
    key="pred_mode",
)

if mode.startswith("Strict"):
    st.caption(
        "Strict mode only shows a result once enough observations are provided to make the estimate meaningful. "
        "Some missing model inputs may still be filled using typical values from the training dataset — and the app "
        "will show what was filled."
    )
else:
    st.caption(
        "Estimate mode always produces a result, even with limited data, by filling missing model inputs using typical "
        "values from the training dataset. Reliability will be lower if many inputs are filled."
    )

st.divider()


# -----------------------------
# Run prediction (shared)
# -----------------------------
def run_prediction(engineered_feats: Dict[str, Any], core_entered: int, core_entered_vars: List[str]) -> None:
    """
    Performs alignment, (optional) strict checks, prediction, missing reporting, reliability,
    and explanation table.
    """
    # Strict check based on core section completeness
    core_counts = section_counts(core_entered_vars)
    ok, reason = strict_minimum_ok(core_counts)

    if mode.startswith("Strict") and not ok:
        st.warning(
            "No result shown in Strict mode.\n\n"
            f"{reason}\n\n"
            f"Currently entered observations: **{core_entered}**."
        )
        st.info(
            "You can either enter a few more observations, or switch to Estimate mode (less reliable)."
        )
        return

    # Align + impute
    X_imp, present_count, missing_features, X_aligned = align_and_impute(engineered_feats, feature_columns, imputer)

    # Predict
    prob = float(model.predict_proba(X_imp)[:, 1][0])
    band = risk_band(prob, thr_medium, thr_high)

    # Counts
    model_total = len(feature_columns)
    imputed_count = len(missing_features)
    core_total = len(starter_vars)

    badge_label, badge_emoji = reliability_badge(core_entered, core_total, imputed_count, model_total)

    # ---------------- Result display ----------------
    st.subheader("Result")

    band_dot = {"LOW": "🟢", "MEDIUM": "🟠", "HIGH": "🔴"}[band]
    st.markdown(f"### {band_dot} {band} RISK")

    st.metric("Estimated mortality risk", f"{prob*100:.2f}%")
    st.caption(f"Model probability: {prob:.6f}")
    st.caption(f"Risk bands: LOW < {thr_medium:.2f}, MEDIUM {thr_medium:.2f}–{thr_high:.3f}, HIGH ≥ {thr_high:.3f}")

    st.divider()

    # ---------------- Reliability ----------------
    st.subheader("Reliability")
    st.markdown(f"**{badge_emoji} {badge_label}**")
    st.write(f"Core observations entered: **{core_entered}/{core_total}**")
    st.write(f"Model inputs filled using training defaults: **{imputed_count}/{model_total}**")

    st.divider()

    # ---------------- Missing filled values (specific) ----------------
    st.subheader("What data was missing (and filled using training defaults)")
    missing_display = build_missing_display(missing_features)
    if len(missing_display) == 0:
        st.write("None — all required model inputs were provided or engineered from your entries.")
    else:
        st.caption(
            "These inputs were not available from the uploaded/entered data, so the model filled them using typical "
            "values from the training dataset. This can reduce reliability."
        )
        for item in missing_display:
            st.write(f"• {item}")

    st.divider()

    # ---------------- Explainable AI (always shown) ----------------
    st.subheader("Why this risk estimate?")
    st.caption(
        "These are the top factors that most increased the model’s estimate for this case. "
        "This is not proof of causation."
    )

    base_model = get_base_model_for_shap(model)
    drivers_df = pd.DataFrame(columns=["feature", "value", "impact"])
    if base_model is not None:
        drivers_df = top_push_drivers(base_model, X_imp, feature_columns, k=5)

    if drivers_df.empty:
        st.write(
            "No clear drivers were available for display. This can happen if the explanation step fails, "
            "or if the estimate is driven mostly by baseline/default fills."
        )
    else:
        # Round for display
        show_df = drivers_df.copy()
        show_df["value"] = show_df["value"].round(4)
        show_df["impact"] = show_df["impact"].round(6)
        show_df = show_df.rename(columns={"feature": "Factor (pushes risk)", "value": "Value used", "impact": "Model impact"})
        st.dataframe(show_df, use_container_width=True, hide_index=True)

    st.divider()

    # ---------------- Export + Technical details (no nested expanders) ----------------
    with st.expander("Export"):
        export_row = X_aligned.copy()
        export_row.insert(0, "predicted_probability", prob)
        export_row.insert(1, "risk_band", band)
        csv_bytes = export_row.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download aligned model inputs (CSV)",
            data=csv_bytes,
            file_name="icu_mortality_inputs_and_prediction.csv",
            mime="text/csv",
        )

    with st.expander("Technical details"):
        st.write("Engineered values available (aligned to the model, pre-imputation):")
        st.dataframe(X_aligned, use_container_width=True)
        st.write("Imputed model inputs (after filling missing values):")
        st.dataframe(X_imp, use_container_width=True)


# -----------------------------
# Upload mode
# -----------------------------
if input_method == "Upload patient file":
    st.subheader("Upload patient file")

    uploaded = st.file_uploader("Upload a single patient .txt file", type=["txt"])

    if uploaded is None:
        st.info("Upload a patient file to get a prediction.")
        st.stop()

    if uploaded.size > 2_000_000:
        st.error("File too large for this demo (max 2MB).")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = Path(tmp.name)

    try:
        long_df = load_patient_long(tmp_path)
        feats = summarise_patient(long_df)

        # Core counts for upload mode are not timepoint-based in this UI,
        # but we can approximate using starter_vars presence.
        core_entered_vars = []
        for v in starter_vars:
            if feats.get(f"{v}_was_measured", 0) == 1:
                core_entered_vars.append(v)
        core_entered = len(core_entered_vars)

        run_prediction(feats, core_entered, core_entered_vars)

    except Exception as e:
        st.error("Something went wrong while predicting.")
        st.exception(e)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


# -----------------------------
# Manual entry mode
# -----------------------------
else:
    st.subheader("Manual entry")
    tp_raw, _ = render_manual_entry_form()

    # Parse after user hits Calculate risk (form submit triggers rerun, but values are in session_state)
    # Detect submit by checking a known flag Streamlit sets? We’ll infer by presence of any entries,
    # and always allow recalculation when values exist.
    # We'll show a "Validate + calculate" button outside the form would be cleaner, but you asked to keep as-is.

    # Collect and parse values from session_state
    parse_errors: List[str] = []
    tp_values: Dict[str, Dict[str, Any]] = {}

    for m in MEASURES:
        tp_values[m.key] = {}
        for tp_key, _ in TIMEPOINTS:
            widget_key = f"{m.key}_{tp_key}"
            raw = st.session_state.get(widget_key, "")

            if m.kind == "bool":
                val, err = parse_bool(str(raw), m)
            else:
                val, err = parse_numeric(str(raw), m)

            if err:
                parse_errors.append(err)
            tp_values[m.key][tp_key] = val

    # Count entered vars
    engineered, core_entered, core_entered_vars = engineer_from_timepoints(tp_values, starter_vars)

    # Show friendly input warnings (clinician-readable)
    if parse_errors:
        st.warning("Some entries couldn’t be used and will be treated as missing:")
        # de-duplicate
        for msg in sorted(set(parse_errors)):
            st.write(f"• {msg}")

    # Only run prediction if anything was entered (otherwise it spams)
    if core_entered == 0:
        st.info("Enter at least one observation, then click Calculate risk.")
        st.stop()

    run_prediction(engineered, core_entered, core_entered_vars)
