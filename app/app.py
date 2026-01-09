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

# SHAP is optional; we will ALWAYS show a readable fallback explanation if SHAP fails
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
# Constants / configuration
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


# Clinician-friendly labels + plausible ranges.
# If your starter_vars include names not here, they'll still work; they'll just appear under "Other".
VAR_META: dict[str, dict[str, Any]] = {
    # Cardiac
    "HR": {"label": "Heart rate", "units": "bpm", "min": 0.0, "max": 250.0, "type": "num"},
    "SysBP": {"label": "Systolic BP", "units": "mmHg", "min": 40.0, "max": 300.0, "type": "num"},
    "DiasBP": {"label": "Diastolic BP", "units": "mmHg", "min": 20.0, "max": 200.0, "type": "num"},
    "MeanBP": {"label": "MAP", "units": "mmHg", "min": 30.0, "max": 200.0, "type": "num"},
    "MAP": {"label": "MAP", "units": "mmHg", "min": 30.0, "max": 200.0, "type": "num"},
    "SBP": {"label": "Systolic BP", "units": "mmHg", "min": 40.0, "max": 300.0, "type": "num"},
    "DBP": {"label": "Diastolic BP", "units": "mmHg", "min": 20.0, "max": 200.0, "type": "num"},

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
    # handle engineered suffixes
    if name == "MechVent_prop_on":
        base = VAR_META.get("MechVent", {}).get("label", "Mechanical ventilation")
        return f"{base} (proportion on)"

    if "_" in name:
        base, suffix = name.split("_", 1)
        base_label = VAR_META.get(base, {}).get("label", base)
        suffix_map = {"mean": "average", "last": "most recent", "was_measured": "recorded"}
        if suffix in suffix_map:
            return f"{base_label} ({suffix_map[suffix]})"
        return f"{base_label} ({suffix.replace('_', ' ')})"

    return VAR_META.get(name, {}).get("label", name)


def bullet_list(items: list[str]) -> None:
    if not items:
        st.write("None.")
        return
    st.markdown("\n".join([f"- {x}" for x in items]))


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

    # kill pandas NAType before sklearn sees it
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
    # CalibratedClassifierCV has calibrated_classifiers_[i].estimator
    if hasattr(model_obj, "calibrated_classifiers_"):
        try:
            cc = model_obj.calibrated_classifiers_[0]
            if hasattr(cc, "estimator"):
                return cc.estimator
            if hasattr(cc, "base_estimator"):
                return cc.base_estimator
        except Exception:
            pass
    # fallback
    if hasattr(model_obj, "estimator"):
        try:
            return model_obj.estimator
        except Exception:
            pass
    return model_obj


def format_push(v: float) -> str:
    arrow = "↑" if v > 0 else ("↓" if v < 0 else "→")
    return f"{arrow} {v:+.4f}"


def try_shap_explain(
    model_obj: Any,
    X_row_imp: pd.DataFrame,
    feature_columns: list[str],
    top_k: int = 8,
) -> tuple[bool, pd.DataFrame | None, str]:
    if not SHAP_AVAILABLE:
        return False, None, "Model explanation isn’t available on this deployment."

    try:
        base_est = get_uncalibrated_estimator(model_obj)

        # TreeExplainer works for RandomForest / tree ensembles
        explainer = shap.TreeExplainer(base_est)
        sv = explainer.shap_values(X_row_imp)

        if isinstance(sv, list):
            sv_pos = sv[1]
        else:
            if getattr(sv, "ndim", 0) == 3:
                sv_pos = sv[:, :, 1]
            else:
                sv_pos = sv

        contrib = np.asarray(sv_pos)[0].astype(float)
        vals = X_row_imp.iloc[0].to_numpy().astype(float)

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
        return True, df, "Model-based explanation generated."

    except Exception:
        return False, None, "Model explanation couldn't be generated here, so a clinician-friendly fallback is shown instead."


def fallback_explanation_from_inputs(
    X_raw_aligned: pd.DataFrame,
    top_k: int = 8,
) -> pd.DataFrame:
    """
    Clinician-friendly fallback: highlight the most extreme (most abnormal) values
    based on the plausibility ranges in VAR_META.
    Not a model explanation; it's an input sanity / “likely drivers” view.
    """
    row = X_raw_aligned.iloc[0]
    items = []

    for feat_name, val in row.items():
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue

        # only attempt to score engineered mean/last variables we know ranges for
        base = feat_name.split("_")[0] if "_" in feat_name else feat_name
        meta = VAR_META.get(base)
        if not meta or meta.get("type") == "binary":
            continue

        vmin = float(meta.get("min", -np.inf))
        vmax = float(meta.get("max", np.inf))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            continue

        mid = (vmin + vmax) / 2.0
        score = abs(float(val) - mid) / (vmax - vmin)  # 0=middle, 0.5=at range edges
        items.append((score, feat_name, float(val), vmin, vmax))

    items.sort(reverse=True, key=lambda x: x[0])
    items = items[:top_k]

    rows = []
    for score, feat_name, val, vmin, vmax in items:
        rows.append(
            {
                "Factor": humanise_feature_name(feat_name),
                "Value used": val,
                "Why highlighted": f"Relatively extreme within plausible range ({vmin:g}–{vmax:g})",
            }
        )

    if not rows:
        rows = [{"Factor": "No usable values entered", "Value used": np.nan, "Why highlighted": "Enter some values to see guidance."}]

    return pd.DataFrame(rows)


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

        # binary vars
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

        # numeric vars
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


def auto_group_vars(starter_vars: list[str]) -> dict[str, list[str]]:
    """
    Dynamic grouping based on name patterns, so BP ALWAYS ends up under Cardiac
    even if your dataset uses slightly different variable names.
    """
    groups = {
        "Respiratory & ventilation": [],
        "Cardiac": [],
        "Blood gas": [],
        "Bloods / labs": [],
        "Other": [],
    }

    for v in starter_vars:
        vl = v.lower()

        # Cardiac patterns (BP + HR)
        if any(k in vl for k in ["hr", "heart"]) and "hco3" not in vl:
            groups["Cardiac"].append(v)
            continue
        if any(k in vl for k in ["bp", "map", "sbp", "dbp", "sys", "dias", "meanbp"]):
            groups["Cardiac"].append(v)
            continue

        # Resp patterns
        if any(k in vl for k in ["resp", "rr", "sao2", "spo2", "fio2", "mechvent", "vent"]):
            groups["Respiratory & ventilation"].append(v)
            continue

        # Blood gas
        if any(k in vl for k in ["ph", "lactate", "hco3", "bicarb", "pco2", "po2", "base"]):
            groups["Blood gas"].append(v)
            continue

        # Bloods / labs
        if any(k in vl for k in ["wbc", "plate", "plt", "creat", "bun", "urea", "glucose", "na", "k", "hct", "mg"]):
            groups["Bloods / labs"].append(v)
            continue

        # Other
        groups["Other"].append(v)

    # Keep a stable order
    for g in groups:
        groups[g] = sorted(groups[g])

    return groups


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="ICU Mortality Demo", layout="centered")

st.title("ICU Mortality Prediction (Demo)")
st.caption("Educational demo only. Not for clinical use.")

with st.expander("What this is (and isn’t)"):
    st.write(
        """
        This demo estimates ICU mortality risk using routine observations and blood results.

        It’s a **portfolio demonstration** of clinical decision-support software.
        It has **not** been validated for real clinical use and must not be used for patient care.
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

page = st.radio(
    "Mode",
    ["Upload patient file (.txt)", "Manual entry"],
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
        st.stop()

    if uploaded.size > MAX_UPLOAD_BYTES:
        st.error("File too large for this demo (max 2MB).")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = Path(tmp.name)

    try:
        long_df = load_patient_long(tmp_path)
        feats = summarise_patient(long_df)

        X_imp, present_count, missing_features, X_raw_aligned = align_and_impute(
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
                clean = [humanise_feature_name(x) for x in missing_features]
                bullet_list(clean)

        # EXPLANATION: NOT OPTIONAL for MEDIUM/HIGH, and NEVER “does nothing”
        if band in ("MEDIUM", "HIGH"):
            st.subheader("Why this score (explainable view)")
            with st.spinner("Generating explanation..."):
                ok, df_shap, msg = try_shap_explain(model, X_imp, feature_columns, top_k=8)

            if ok and df_shap is not None:
                st.caption("Model-based explanation (how the model pushed the risk up/down). Not proof of causality.")
                st.dataframe(df_shap, use_container_width=True)
            else:
                st.caption(msg)
                st.caption("Fallback explanation: highlights the most extreme values entered (clinician-friendly).")
                df_fb = fallback_explanation_from_inputs(X_raw_aligned, top_k=8)
                st.dataframe(df_fb, use_container_width=True)

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
        st.stop()

    st.write(
        """
        Enter values at **12 hours ago**, **6 hours ago**, and **Now**.

        The model uses engineered values:
        - **Average** of what you entered
        - **Most recent** value (Now → 6h → 12h)
        """
    )

    GROUPS = auto_group_vars(starter_vars)
    inputs: dict[str, dict[str, Any]] = {v: {} for v in starter_vars}

    submitted = False
    with st.form("manual_form", clear_on_submit=False):
        for group_name, var_list in GROUPS.items():
            if not var_list:
                continue

            st.markdown(f"## {group_name}")

            for var in var_list:
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

                        # range warning is built into number_input via min/max
                        help_text = f"Please enter a value between {min_v:g} and {max_v:g}"
                        if units:
                            help_text += f" {units}"

                        step = 0.01 if (max_v - min_v) <= 1 else (0.1 if (max_v - min_v) <= 20 else 1.0)

                        inputs[var][tp_key] = cols[idx].number_input(
                            tp_label,
                            min_value=min_v,
                            max_value=max_v,
                            value=None,
                            step=step,
                            key=f"{var}_{tp_key}_num",
                            help=help_text,
                        )

                st.write("")

            st.divider()

        submitted = st.form_submit_button("Calculate risk")

    if submitted:
        try:
            entered_vars = count_variables_entered(inputs, starter_vars)

            raw_engineered = manual_inputs_to_engineered_features(inputs, starter_vars)
            X_imp, present_count, missing_features, X_raw_aligned = align_and_impute(
                raw_engineered, feature_columns, imputer
            )

            prob = float(model.predict_proba(X_imp)[:, 1][0])
            band = risk_band(prob, thr_low, thr_high)

            st.subheader("Result")
            show_risk_line(band)
            st.metric("Predicted mortality risk (probability)", f"{prob:.6f}")

            st.subheader("Data completeness")
            st.write(f"Clinical variables entered: **{entered_vars}/{len(starter_vars)}**")

            st.subheader("Missing data items")
            st.write(f"Missing items used by the model (imputed): **{len(missing_features)}**")
            if missing_features:
                with st.expander("Show missing items"):
                    clean = [humanise_feature_name(x) for x in missing_features]
                    bullet_list(clean)

            # EXPLANATION: NOT OPTIONAL for MEDIUM/HIGH, and NEVER “does nothing”
            if band in ("MEDIUM", "HIGH"):
                st.subheader("Why this score (explainable view)")
                with st.spinner("Generating explanation..."):
                    ok, df_shap, msg = try_shap_explain(model, X_imp, feature_columns, top_k=8)

                if ok and df_shap is not None:
                    st.caption("Model-based explanation (how the model pushed the risk up/down). Not proof of causality.")
                    st.dataframe(df_shap, use_container_width=True)
                else:
                    st.caption(msg)
                    st.caption("Fallback explanation: highlights the most extreme values entered (clinician-friendly).")
                    df_fb = fallback_explanation_from_inputs(X_raw_aligned, top_k=8)
                    st.dataframe(df_fb, use_container_width=True)

        except Exception as e:
            st.error("Something went wrong while calculating risk.")
            with st.expander("Technical details (for debugging)"):
                st.exception(e)

with st.expander("Technical details (for reviewers / developers)"):
    st.write(f"Bundle path: `{bundle_path.as_posix()}`")
    st.write(f"Features expected: {len(feature_columns)}")
    st.write(f"Starter vars in bundle: {len(starter_vars) if isinstance(starter_vars, list) else 'missing'}")
    st.write(f"SHAP available: {SHAP_AVAILABLE}")
