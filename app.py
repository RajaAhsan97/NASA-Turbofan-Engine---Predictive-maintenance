"""
Predictive Maintenance Dashboard — NASA C-MAPSS Turbofan Engines
==================================================================
Companion Streamlit app for the "predictive_maintenance_nasa_turbofan.ipynb" webinar notebook.

Run:
    pip install streamlit plotly pandas numpy scikit-learn joblib langchain langchain-google-genai google-generativeai
    export GOOGLE_API_KEY="your-key-here"          # optional, enables live GenAI briefings
    streamlit run app.py

Expects, relative to this file:
    ./data/train_FD001.txt, test_FD001.txt, RUL_FD001.txt   (raw C-MAPSS files)
    ./models/best_rul_model.pkl, scaler.pkl, feature_cols.pkl, feature_sensors.pkl
        (produced by running the notebook's Sections 0-6 at least once)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

import langchain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import streamlit as st

# ----------------------------------------------------------------------------
# Page config & style
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Turbofan Predictive Maintenance",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {background-color: #0e1117; border: 1px solid #262730; border-radius: 10px; padding: 14px;}
    .status-healthy  {color:#22c55e; font-weight:700;}
    .status-watch    {color:#eab308; font-weight:700;}
    .status-warning  {color:#f97316; font-weight:700;}
    .status-critical {color:#ef4444; font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)

llm_model = "gemini-3-flash-preview"

# Load local .env  - locally
load_dotenv()
llm_api_key = os.getenv("GOOGLE_API_KEY")

# If running on Streamlit Cloud, use Secrets
if not llm_api_key:
    llm_api_key = st.secrets["GOOGLE_API_KEY"]

DATA_DIR = "data/archive/CMaps"
MODEL_DIR = "models"
DATASET_ID = "FD001"
RUL_CLIP = 125

INDEX_COLS = ["unit_number", "time_in_cycles"]
SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [
    'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30',        # Temperatures & Pressures
    'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc',       # Rotational Speeds
    'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32' # Others
]
COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

SENSOR_DESCRIPTIONS = {
    "sensor_1": "Total temperature at fan inlet (T2)",
    "sensor_2": "Total temperature at LPC outlet (T24)",
    "sensor_3": "Total temperature at HPC outlet (T30)",
    "sensor_4": "Total temperature at LPT outlet (T50)",
    "sensor_5": "Pressure at fan inlet (P2)",
    "sensor_6": "Total pressure in bypass-duct (P15)",
    "sensor_7": "Total pressure at HPC outlet (P30)",
    "sensor_8": "Physical fan speed (Nf)",
    "sensor_9": "Physical core speed (Nc)",
    "sensor_10": "Engine pressure ratio (epr)",
    "sensor_11": "Static pressure at HPC outlet (Ps30)",
    "sensor_12": "Ratio of fuel flow to Ps30 (phi)",
    "sensor_13": "Corrected fan speed (NRf)",
    "sensor_14": "Corrected core speed (NRc)",
    "sensor_15": "Bypass Ratio (BPR)",
    "sensor_16": "Burner fuel-air ratio (farB)",
    "sensor_17": "Bleed Enthalpy (htBleed)",
    "sensor_18": "Demanded fan speed (Nf_dmd)",
    "sensor_19": "Demanded corrected fan speed (PCNfR_dmd)",
    "sensor_20": "HPT coolant bleed (W31)",
    "sensor_21": "LPT coolant bleed (W32)",
}

MAINTENANCE_PROMPT_TEMPLATE = """You are an experienced aerospace maintenance engineer assistant.
You will be given a structured health snapshot for a turbofan engine derived from sensor telemetry
and a machine learning Remaining Useful Life (RUL) model.

Engine Health Snapshot (JSON):
{snapshot}

Write a concise maintenance briefing with these sections:
1. **Health Status** — one-line verdict (Healthy / Watch / Warning / Critical) based on the current RUL.
2. **Key Drivers** — plain-language explanation of which sensors are deviating from healthy baseline and what that physically implies.
3. **Trend Read** — what the sensor deviations suggest about the pace of degradation.
4. **Recommended Actions** — 2-4 concrete, prioritized maintenance actions with suggested timing.

Keep it under 200 words, professional tone, no markdown tables.
"""


# ----------------------------------------------------------------------------
# Cached data / model loading
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "best_rul_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    feature_sensors = joblib.load(os.path.join(MODEL_DIR, "feature_sensors.pkl"))
    return model, scaler, feature_cols, feature_sensors


@st.cache_data(show_spinner="Loading C-MAPSS data...")
def load_data(dataset_id, _feature_cols_base):
    train_path = os.path.join(DATA_DIR, f"train_{dataset_id}.txt")
    test_path = os.path.join(DATA_DIR, f"test_{dataset_id}.txt")
    rul_path = os.path.join(DATA_DIR, f"RUL_{dataset_id}.txt")

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLS)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLS)
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"])

    max_cycle = train_df.groupby("unit_number")["time_in_cycles"].transform("max")
    train_df["RUL"] = max_cycle - train_df["time_in_cycles"]
    train_df["RUL_clipped"] = train_df["RUL"].clip(upper=RUL_CLIP)
    return train_df, test_df, rul_df


def scale_and_engineer(df, scaler, feature_sensors, feature_cols_base, window=5):
    df = df.copy()
    df[feature_cols_base] = scaler.transform(df[feature_cols_base])
    df = df.sort_values(["unit_number", "time_in_cycles"])
    grouped = df.groupby("unit_number")[feature_sensors]
    roll_mean = grouped.rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
    roll_std = grouped.rolling(window=window, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
    roll_mean.columns = [f"{c}_rmean{window}" for c in feature_sensors]
    roll_std.columns = [f"{c}_rstd{window}" for c in feature_sensors]
    return pd.concat([df, roll_mean, roll_std], axis=1)


def health_status(rul):
    if rul < 20:
        return "Critical", "status-critical"
    if rul < 50:
        return "Warning", "status-warning"
    if rul < 100:
        return "Watch", "status-watch"
    return "Healthy", "status-healthy"


def rule_based_fallback_briefing(snapshot):
    rul = snapshot["current_rul_cycles"]
    status, _ = health_status(rul)
    top = ", ".join(s["description"] for s in snapshot["top_deviating_sensors"][:3])
    return (
        f"**Health Status:** {status}. Current RUL ~{rul} cycles. "
        f"Top deviating readings vs healthy baseline: {top}. "
        f"**Recommended:** schedule inspection within {max(int(rul * 0.3), 1)} cycles "
        f"and monitor these sensors closely over the next few runs."
    )


def get_maintenance_briefing(snapshot, api_key):
    if not api_key:
        return rule_based_fallback_briefing(snapshot) + "\n\n*(Set GOOGLE_API_KEY to enable live Gemini briefings.)*"
    try:
        llm = ChatGoogleGenerativeAI(model=llm_model, google_api_key=api_key, temperature=0.3)
        prompt = PromptTemplate.from_template(MAINTENANCE_PROMPT_TEMPLATE)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"snapshot": json.dumps(snapshot, indent=2)})
    except Exception as e:
        return f"*(GenAI call failed: {e})*\n\n" + rule_based_fallback_briefing(snapshot)


def build_snapshot(unit_number, current_row, engine_history, feature_sensors):
    # Build a compact health snapshot using only the current predicted RUL and recent sensor
    # deviation from a healthy (early-life) baseline for this engine — no forecasting required.
    current_rul = float(current_row["RUL_pred"])
    recent_mean = engine_history[feature_sensors].tail(5).mean()
    early_mean = engine_history[feature_sensors].head(5).mean()
    deviation = (recent_mean - early_mean).abs().sort_values(ascending=False)
    top_sensors = deviation.head(7).index.tolist()
    return {
        "unit_number": int(unit_number),
        "current_cycle": int(current_row["time_in_cycles"]),
        "current_rul_cycles": round(current_rul, 1),
        "top_deviating_sensors": [
            {"sensor": s, "description": SENSOR_DESCRIPTIONS.get(s, s), "deviation": round(float(deviation[s]), 4)}
            for s in top_sensors
        ],
    }


# ----------------------------------------------------------------------------
# App body
# ----------------------------------------------------------------------------
st.title("🛠️ Turbofan Engine Predictive Maintenance")
st.caption("NASA C-MAPSS · RUL Prediction · Sensor Forecasting · GenAI Maintenance Assistant")

missing = [f for f in ["best_rul_model.pkl", "scaler.pkl", "feature_cols.pkl", "feature_sensors.pkl"]
           if not os.path.exists(os.path.join(MODEL_DIR, f))]
if missing:
    st.error(
        "Missing model artifacts: " + ", ".join(missing) +
        ". Run the companion notebook's Sections 0-5 first to train and save the model."
    )
    st.stop()

model, scaler, feature_cols, feature_sensors = load_artifacts()
feature_cols_base = SETTING_COLS + feature_sensors

try:
    train_df, test_df, rul_df = load_data(DATASET_ID, feature_cols_base)
except FileNotFoundError:
    st.error(f"Raw data files not found in ./{DATA_DIR}/. Add train_{DATASET_ID}.txt, "
             f"test_{DATASET_ID}.txt, RUL_{DATASET_ID}.txt.")
    st.stop()

test_df = test_df.copy()
test_df["engine_id"] = test_df["unit_number"]
train_fe = scale_and_engineer(train_df, scaler, feature_sensors, feature_cols_base)
test_fe = scale_and_engineer(test_df, scaler, feature_sensors, feature_cols_base)

test_fe["RUL_pred"] = np.clip(model.predict(test_fe[feature_cols]), 0, None)
test_last = test_fe.groupby("unit_number").tail(1).reset_index(drop=True)

# ---- Sidebar ----
st.sidebar.header("⚙️ Controls")
#api_key_input = st.sidebar.text_input("Gemini API Key (optional)", value=os.environ.get("GOOGLE_API_KEY", ""), type="password")
selected_unit = st.sidebar.selectbox("Select Engine", sorted(test_last["unit_number"].unique()))
st.sidebar.markdown("---")
st.sidebar.metric("Model", type(model).__name__)
st.sidebar.metric("Fleet size (test)", test_last["unit_number"].nunique())
st.sidebar.caption("Dataset: C-MAPSS " + DATASET_ID)

# ---- Fleet Overview ----
tab_fleet, tab_engine, tab_assistant, tab_model = st.tabs(
    ["📊 Fleet Overview", "🔧 Engine Drill-Down", "🤖 GenAI Assistant", "📋 Model Card"]
)

with tab_fleet:
    col1, col2, col3, col4 = st.columns(4)
    n_critical = (test_last["RUL_pred"] < 20).sum()
    n_warning = ((test_last["RUL_pred"] >= 20) & (test_last["RUL_pred"] < 50)).sum()
    n_watch = ((test_last["RUL_pred"] >= 50) & (test_last["RUL_pred"] < 100)).sum()
    n_healthy = (test_last["RUL_pred"] >= 100).sum()
    col1.metric("🔴 Critical", int(n_critical))
    col2.metric("🟠 Warning", int(n_warning))
    col3.metric("🟡 Watch", int(n_watch))
    col4.metric("🟢 Healthy", int(n_healthy))

    fig = px.histogram(
        test_last, x="RUL_pred", nbins=30, title="Predicted RUL Distribution Across Fleet",
        labels={"RUL_pred": "Predicted RUL (cycles)"}, color_discrete_sequence=["#3b82f6"],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("At-Risk Engines (lowest predicted RUL)")
    risk_table = test_last.sort_values("RUL_pred")[["unit_number", "time_in_cycles", "RUL_pred"]].head(10)
    risk_table.columns = ["Engine", "Current Cycle", "Predicted RUL"]
    st.dataframe(risk_table, use_container_width=True, hide_index=True)

with tab_engine:
    eng_row = test_last[test_last.unit_number == selected_unit].iloc[0]
    status, css_class = health_status(eng_row["RUL_pred"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Cycle", int(eng_row["time_in_cycles"]))
    c2.metric("Predicted RUL", f"{eng_row['RUL_pred']:.1f} cycles")
    c3.markdown(f"**Status:** <span class='{css_class}'>{status}</span>", unsafe_allow_html=True)

    st.markdown("#### Sensor Trends")
    eng_history = test_df[test_df.unit_number == selected_unit].sort_values("time_in_cycles")
    default_sensors = feature_sensors[:4]
    chosen_sensors = st.multiselect(
        "Sensors to display", options=feature_sensors,
        default=default_sensors, format_func=lambda s: f"{s} — {SENSOR_DESCRIPTIONS.get(s, '')}",
    )
    if chosen_sensors:
        fig2 = go.Figure()
        for s in chosen_sensors:
            fig2.add_trace(go.Scatter(x=eng_history["time_in_cycles"], y=eng_history[s], mode="lines", name=s))
        fig2.update_layout(title=f"Engine {selected_unit} — Raw Sensor Trends", xaxis_title="Cycle", yaxis_title="Reading")
        st.plotly_chart(fig2, use_container_width=True)

with tab_assistant:
    st.markdown("Ask the assistant for a maintenance briefing on the currently selected engine, "
                f"**Engine {selected_unit}**.")
    if st.button("Generate Maintenance Briefing", type="primary"):
        with st.spinner("Analyzing engine health and generating briefing..."):
            current_row = test_last[test_last.unit_number == selected_unit].iloc[0]
            engine_history = test_fe[test_fe.unit_number == selected_unit].sort_values("time_in_cycles")
            snapshot = build_snapshot(selected_unit, current_row, engine_history, feature_sensors)
            briefing = get_maintenance_briefing(snapshot, llm_api_key)
        with st.expander("Health Snapshot (JSON sent to LLM)"):
            st.json(snapshot)
        st.markdown("### 📋 Maintenance Briefing")
        st.markdown(briefing)

    st.markdown("---")
    st.caption("Powered by LangChain. ")

with tab_model:
    st.markdown(f"**AlgoPrithm:** `{type(model).__name__}`")
    st.markdown(f"**Features used:** {len(feature_cols)}")
    st.markdown(f"**RUL clipping cap:** {RUL_CLIP} cycles")

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)
        fig3 = px.bar(importances[::-1], orientation="h", title="Top 15 Feature Importances",
                      labels={"value": "Importance", "index": "Feature"})
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("#### Actual vs Predicted RUL (Test Set)")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(y=rul_df["RUL"].clip(upper=RUL_CLIP), mode="markers", name="Actual"))
    fig4.add_trace(go.Scatter(y=test_last["RUL_pred"], mode="markers", name="Predicted"))
    fig4.update_layout(xaxis_title="Test Engine (index)", yaxis_title="RUL (cycles)")
    st.plotly_chart(fig4, use_container_width=True)
