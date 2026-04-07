import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from utils import apply_saas_theme, dashboard_header

# ── Setup ──
BACKEND = "http://localhost:8000"
st.set_page_config(page_title="Evaluation / Benchmarks", layout="wide")
apply_saas_theme()

dashboard_header("Model Performance", "ML Ensemble Heuristic Evaluation")

try:
    data = requests.get(f"{BACKEND}/model-comparison", timeout=3).json()
except Exception as e:
    st.error(f"Engine Unreachable: {e}")
    st.stop()

if "error" in data:
    st.info(f"Analysis Pending: {data['error']}")
    st.stop()

# ── Performance Matrix ──
df = pd.DataFrame(data).T
df.index.name = "Model Architecture"

# Separate metrics from metadata/nested dicts for the table
# Ensure we ONLY have numeric columns to prevent highlighting crashes
numeric_cols = ["accuracy", "precision", "recall", "f1_score", "inference_ms"]
table_df = df[df.columns.intersection(numeric_cols)].apply(pd.to_numeric, errors="coerce")
import numpy as np
table_df = table_df.select_dtypes(include=[np.number])
table_df.columns = [c.title() for c in table_df.columns]

st.subheader("Performance Matrix")
st.dataframe(table_df.style.highlight_max(axis=0, color="rgba(0, 112, 243, 0.2)"), width="stretch")

# ── Visual Benchmarking (Plotly) ──
st.divider()
st.subheader("Accuracy & F1 Benchmark")

# High-contrast technical Plotly bars
fig = go.Figure()

fig.add_trace(go.Bar(
    x=table_df.index, y=table_df["Accuracy"],
    name="Accuracy", marker_color="#0ea5e9",
    text=table_df["Accuracy"].round(3), textposition='auto',
))

fig.add_trace(go.Bar(
    x=table_df.index, y=table_df["F1_score"],
    name="F1 Score", marker_color="#64748b",
    marker_line_color="#0ea5e9", marker_line_width=1,
    text=table_df["F1_score"].round(3), textposition='auto',
))

fig.update_layout(
    barmode='group',
    xaxis=dict(showgrid=False, title=""),
    yaxis=dict(showgrid=False, range=[0, 1.1], title="Score"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=0, r=0, t=50, b=0),
    height=400
)
st.plotly_chart(fig, width="stretch", theme="streamlit", config={'displayModeBar': False})
