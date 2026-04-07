import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from utils import apply_saas_theme, dashboard_header, format_local_time

# ── Setup ──
BACKEND = "http://localhost:8000"
st.set_page_config(page_title="Analysis / XAI", layout="wide")
apply_saas_theme()

dashboard_header("Diagnostic Heuristics", "Explainable AI (XAI) Attribution Analysis")

try:
    # 1. Fetch recent telemetry to select from
    preds = requests.get(f"{BACKEND}/predictions?limit=50", timeout=3).json()
except Exception as e:
    st.error(f"Engine Unreachable: {e}")
    st.stop()

if not preds:
    st.info("Monitoring Active: Waiting for incidents")
    st.stop()

# ── Investigation Hub ──
options = {
    f"Log ID: {p['id']} / {format_local_time(p['timestamp'])} / {p['source_ip']} / {p['attack_type'].title()}": p['id']
    for p in preds
}
selected_label = st.selectbox("Select incident log to investigate:", list(options.keys()))
selected_id = options[selected_label]

# 2. Dynamic Fetch
try:
    detail = requests.get(f"{BACKEND}/prediction/{selected_id}", timeout=3).json()
except Exception as e:
    st.error(f"Log Fetch Error: {e}")
    st.stop()

# ── Visualization ──
st.divider()
c1, c2 = st.columns([0.35, 0.65])

def style_severity(val):
    v = str(val).upper()
    if v == "CRITICAL": return "color: #DC2626; font-weight: 600;" # Red
    if v == "HIGH": return "color: #EA580C; font-weight: 600;"    # Orange
    if v == "MEDIUM": return "color: #CA8A04;"                    # Yellow
    if v == "LOW": return "color: #6B7280;"                       # Gray
    return "color: #16A34A;"                                      # Green

with c1:
    st.subheader("Metadata")
    st.markdown(f"""
<div style="background-color: var(--secondary-background-color); padding: 1rem; border: 1px solid var(--secondary-background-color); border-radius: 4px;">
<p style="margin:0; font-size:0.7rem; color:var(--text-color); opacity:0.7;">UUID</p>
<p style="margin:0; font-size:1.1rem; color:var(--text-color); margin-bottom:1rem;">NETWATCH-{detail['id']:05}</p>
<p style="margin:0; font-size:0.7rem; color:var(--text-color); opacity:0.7;">Traffic Type</p>
<p style="margin:0; font-size:1.1rem; color:var(--text-color); margin-bottom:1rem;">{detail['attack_type'].title()}</p>
<p style="margin:0; font-size:0.7rem; color:var(--text-color); opacity:0.7;">Severity</p>
<p style="margin:0; font-size:1.1rem; {style_severity(detail['severity'])} margin-bottom:1rem;">{detail['severity'].title()}</p>
<p style="margin:0; font-size:0.7rem; color:var(--text-color); opacity:0.7;">Anomaly Index</p>
<p style="margin:0; font-size:1.1rem; color:var(--text-color);">{round(detail['anomaly_score'], 4)}</p>
</div>
""", unsafe_allow_html=True)

with c2:
    st.subheader("Attribution Matrix")
    shap_data = detail.get("shap_values", {})
    if shap_data:
        # Sort by absolute impact for premium diagnostics look
        shap_series = pd.Series(shap_data).sort_values(key=abs, ascending=False).head(15)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=shap_series.values[::-1],
            y=shap_series.index[::-1],
            orientation='h',
            # Red for attack impact, Blue for normal impact
            marker_color=["#ef4444" if v > 0 else "#0ea5e9" for v in shap_series.values[::-1]],
            marker_line_width=0
        ))
        
        fig.update_layout(
            xaxis=dict(showgrid=False, title="Impact Contribution"),
            yaxis=dict(showgrid=False, title=""),
            margin=dict(l=0, r=0, t=10, b=0),
            height=450
        )
        st.plotly_chart(fig, width="stretch", theme="streamlit", config={'displayModeBar': False})
        st.caption("Red: Impact toward Attack | Blue: Impact toward Normal")
    else:
        st.warning("Heuristic data stream empty.")
