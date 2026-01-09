"""
Streamlit demo app (educational).

Run locally (optional):
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

# Optional explainability
try:
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False


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

RISK_DOT = {"LOW": "🟢", "MEDIUM": "🟠", "HIGH": "🔴"}

# Clinician-friendly labels + units + plausible ranges
VAR_META: dict[str, dict[str, Any]] = {
    # Cardiac
    "HR": {"label": "Heart rate", "units": "bpm", "min": 0.0, "max": 250.0, "type": "num"},
    "SysBP": {"label": "Systolic BP", "units": "mmHg", "min": 0.0, "max": 300.0, "type": "num"},
    "DiasBP": {"label": "Diastolic BP", "units": "mmHg", "min": 0.0, "max": 200.0, "type": "num"},
    "MeanBP": {"label": "MAP", "units": "mmHg", "min": 0.0, "max": 200.0, "type": "num"},

    # Respiratory & ventilation
    "RespRate": {"label": "Respiratory rate", "units": "breaths/min", "min": 0.0, "max": 80.0, "type": "num"},
    "SaO2": {"label": "Oxygen saturation", "units": "%", "min": 0.0, "max": 100.0, "type": "num"},
    "FiO2": {"label": "FiO₂", "units": "fraction (0.21–1.00)", "min": 0.21, "max": 1.0, "type": "fio2_fraction"},
    "MechVent": {"label": "Invasive mechanical ventilation", "units": "Yes/No", "type": "binary"},

    # Other observations
    "Temp": {"label": "Temperature", "units": "°C", "min": 25.0, "max": 45.0, "type": "num"},

    # Blood gas
    "pH": {"label": "pH", "units": "", "min": 6.6, "max": 7.8, "type": "num"},
    "HCO3": {"label": "Bicarbonate (HCO₃⁻)", "units": "mmol/L", "min": 0.0, "max": 60.0, "type": "num"},
    "Lactate": {"label": "Lactate", "units": "mmol/L", "min": 0.0, "max": 30.0, "type": "num"},

    # Bloods / labs
    "Glucose": {"label": "Glucose", "units": "mmol/L", "min": 0.0, "max": 60.0, "type": "num"},
    "Creatinine": {"label": "Creatinine", "units": "µmol/L", "min": 0.0, "max": 2000.0, "type": "num"},
    "BUN": {"label": "Urea (BUN)", "units": "mmol/L", "min": 0.0, "max": 80.0, "type": "num"},
    "WBC": {"label": "White cell count", "units": "x10⁹/L", "min": 0.0, "max": 200.0, "type": "num"},
    "Platelets": {"label": "Platelets", "units": "x10⁹/L", "min": 0.0, "max": 2000.0, "type": "num"},
    "Na": {"label": "Sodium (Na⁺)", "units": "mmol/L", "min": 80.0, "max": 200.0, "type": "num"},
    "K": {"label": "Potassium (K⁺)", "units": "mmol/L", "min": 1.0, "max": 10.0, "type": "num"},
    "Hct": {"label": "Haematocrit (Hct)", "units": "fraction (0–1)", "min": 0.0, "max": 1.0, "type": "num"},
    "Mg": {"label": "Magnesium (Mg²⁺)", "units": "mmol/L", "min": 0.0, "max": 5.0, "type": "num"},
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
    if name == "MechVent_prop_on":
        base = VAR_META.get("MechVent", {}).get("label", "Mechanical ventilation")
        return f"{base} (proportion on)"

    if "_" in name:
        base, suffix = name.split("_", 1)
        base_label = VAR_META.get(base, {}).get("label", base)
        suffix_map = {
            "mean": "average",
            "last": "most recent",
            "was_measured": "recorded (yes/no)",
        }
        if suffix in suffix_map:
            return f"{base_label} ({suffix_map[suffix]})"
        return f"{base_label} ({suffix.replace('_', ' ')})"

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

    # Prevent pandas NAType leaking into sklearn
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


def show_risk_line(band: str) -> None:
    st.markdown(f"### {RISK_DOT[band]} **{band} RISK**")


def get_uncalibrated_estimator(model_obj: Any) -> Any:
    if hasattr(model_obj, "calibrated_classifiers_"):
        try:
            cc = model_obj.calibrated_classifiers_[0]
            if hasattr(cc, "estimator"):
                return cc.estimator
            if hasattr(cc, "base_estimator"):
                return cc.base_estimator
        except Exception:
            pass

    if hasattr(model_obj, "estimator"):
        try:
            return model_obj.estimator
        except Exception:
            pass

    return model_obj


def format_push(v: float) -> str:
    """
    Convert SHAP contributions into clinician-friendly 'push' text.
    Positive -> pushes risk up; negative -> pushes risk down.
    """
    arrow = "↑" if v > 0 else ("↓" if v < 0 else "→")
    return f"{arrow} {v:+.4f}"


def try_shap_explain(
    model_obj: Any,
    X_row: pd.DataFrame,
    feature_columns: list[str],
    top_k: int = 8,
) -> tuple[bool, str, pd.DataFrame | None]:
    if not SHAP_AVAILABLE:
        return False, "Explainability isn't available in this environment (SHAP not installed).", None

    try:
        base_est = get_uncalibrated_estimator(model_obj)

        explainer = shap.TreeExplainer(base_est)
        sv = explainer.shap_values(X_row)

        if isinstance(sv, list):
            sv_pos = sv[1]
        else:
            if getattr(sv, "ndim", 0) == 3:
                sv_pos = sv[:, :, 1]
            else:
                sv_pos = sv

        contrib = np.asarray(sv_pos)[0].astype(float)
        vals = X_row.iloc[0].to_numpy().astype(float)

        order = np.argsort(np.abs(contrib))[::-1][:top_k]

        rows = []
        for i in order:
            fname = feature_columns[i]
            rows.append(
                {
                    "Factor": humanise_feature_name(fname),
                    "Value used by model": float(vals[i]),
                    "Pushes risk (↑ / ↓)": format_push(float(contrib[i])),
                }
            )

        df = pd.DataFrame(rows)
        return True, "Explanation generated.", df

    except Exception:
        return (
            False,
            "This model/environment couldn't generate an explanation right now. "
            "The prediction still works — explanation is optional.",
            None,
        )


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


def count_engineered_items_available(raw_engineered: dict, starter_vars: list[str]) -> tuple[int, int]:
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
        This demo estimates ICU mortality risk using a small set of routine observations and blood results.

        It’s a **portfolio demonstration** of how clinical decision-support software could work.
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

                X_imp, present_count, missing_features, X_aligned = align_and_impute(
                    feats, feature_columns, imputer
                )

                prob = float(model.predict_proba(X_imp)[:, 1][0])
                band = risk_band(prob, thr_low, thr_high)

                st.subheader("Result")
                show_risk_line(band)
                st.metric("Predicted mortality risk (probability)", f"{prob:.6f}")

                st.subheader("Missing data items")
                st.write(f"Missing items used by the model (imputed): **{len(missing_features)}**")
                if missing_features:
                    with st.expander("Show missing items"):
                        st.write([humanise_feature_name(x) for x in missing_features])

                if band in ("MEDIUM", "HIGH"):
                    st.subheader("Why this result?")
                    choice = st.selectbox(
                        "Explanation",
                        ["Off", "Show key contributing factors (explainable AI)"],
                        index=0,
                    )

                    if choice != "Off":
                        ok, msg, df = try_shap_explain(model, X_imp, feature_columns, top_k=8)
                        if ok and df is not None:
                            st.caption(
                                "These factors are what the model *used* to form the score. "
                                "They are not proof of causality."
                            )
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.info(msg)

            except Exception as e:
                st.error("Something went wrong while predicting from the uploaded file.")
                with st.expander("Technical details (for debugging)"):
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
    if starter_vars is None or not isinstance(starter_vars, list) or len(starter_vars) == 0:
        st.warning("Manual entry is unavailable because this model bundle does not include `starter_vars`.")
    else:
        st.write(
            """
            Enter values at **12 hours ago**, **6 hours ago**, and **Now**.

            The model uses engineered values:
            - **Average** of the values you entered
            - **Most recent** value (Now → 6h → 12h)
            """
        )

        GROUPS = {
            "Respiratory & ventilation": ["RespRate", "SaO2", "FiO2", "MechVent"],
            "Cardiac": ["HR", "SysBP", "DiasBP", "MeanBP"],
            "Blood gas": ["pH", "HCO3", "Lactate"],
            "Bloods / labs": ["Glucose", "Creatinine", "BUN", "WBC", "Platelets", "Na", "K", "Hct", "Mg"],
            "Other": ["Temp"],
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

                            help_text = f"Please enter a value between {min_v:g} and {max_v:g}"
                            if units:
                                help_text += f" {units}"

                            inputs[var][tp_key] = cols[idx].number_input(
                                tp_label,
                                min_value=min_v,
                                max_value=max_v,
                                value=None,
                                step=0.01 if (max_v - min_v) <= 1 else (0.1 if (max_v - min_v) <= 20 else 1.0),
                                key=f"{var}_{tp_key}_num",
                                help=help_text,
                            )

                    st.write("")

                st.divider()

            if extras:
                st.markdown("## Other variables (used by this model)")
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
                                step=0.1,
                                key=f"{var}_{tp_key}_num_extra",
                            )
                    st.write("")
                st.divider()

            submitted = st.form_submit_button("Calculate risk")

        if submitted:
            try:
                entered_vars = count_variables_entered(inputs, starter_vars)

                raw_engineered = manual_inputs_to_engineered_features(inputs, starter_vars)
                eng_present, eng_total = count_engineered_items_available(raw_engineered, starter_vars)

                X_imp, _present_count, missing_features, _X_aligned = align_and_impute(
                    raw_engineered, feature_columns, imputer
                )

                prob = float(model.predict_proba(X_imp)[:, 1][0])
                band = risk_band(prob, thr_low, thr_high)

                st.subheader("Result")
                show_risk_line(band)
                st.metric("Predicted mortality risk (probability)", f"{prob:.6f}")

                st.subheader("What the model actually used")
                st.write(
                    f"- Clinical variables entered: **{entered_vars}/{len(starter_vars)}**\n"
                    f"- Engineered values created (average + most recent): **{eng_present}/{eng_total}**"
                )

                with st.expander("What does “engineered values” mean?"):
                    st.write(
                        """
                        The model doesn’t take raw “12h/6h/now” directly.
                        For each variable it creates:
                        - **Average** = average of the values you entered
                        - **Most recent** = the latest value available (Now → 6h → 12h)

                        For ventilation:
                        - **MechVent** is treated as Yes=1 / No=0, and the average becomes a rough “proportion on”.
                        """
                    )

                st.subheader("Missing data items")
                st.write(f"Missing items used by the model (imputed): **{len(missing_features)}**")
                if missing_features:
                    with st.expander("Show missing items"):
                        st.write([humanise_feature_name(x) for x in missing_features])

                if band in ("MEDIUM", "HIGH"):
                    st.subheader("Why this result?")
                    choice = st.selectbox(
                        "Explanation",
                        ["Off", "Show key contributing factors (explainable AI)"],
                        index=0,
                    )

                    if choice != "Off":
                        ok, msg, df = try_shap_explain(model, X_imp, feature_columns, top_k=8)
                        if ok and df is not None:
                            st.caption(
                                "These factors are what the model *used* to form the score. "
                                "They are not proof of causality."
                            )
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.info(msg)

            except Exception as e:
                st.error("Something went wrong while calculating risk.")
                with st.expander("Technical details (for debugging)"):
                    st.exception(e)

with st.expander("Technical details (for reviewers / developers)"):
    st.write(f"Bundle path: `{bundle_path.as_posix()}`")
    st.write(f"Features expected: {len(feature_columns)}")
    st.write(f"Starter vars in bundle: {len(starter_vars) if isinstance(starter_vars, list) else 'missing'}")
    st.write(f"Explainable AI available: {SHAP_AVAILABLE}")
