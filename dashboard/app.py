import streamlit as st
from utils import apply_saas_theme, dashboard_header
import requests

# ── Setup ──
BACKEND = "http://localhost:8000"
st.set_page_config(
    page_title="Netwatch IDS",
    layout="wide",
    initial_sidebar_state="expanded"
)
apply_saas_theme()

# ── Sidebar ──
with st.sidebar:
    st.markdown("### Infrastructure")
    st.caption("Node: Edge 01")
    st.caption("SSL/TLS Enabled")
    st.markdown("---")
    st.markdown("### Engine")
    st.caption("XGBoost Ensemble")
    st.caption("SHAP XAI Layer")

# ── Main Content ──
dashboard_header("System Overview", "Network Integrity and Threat Intelligence Interface")

# Top Level Telemetry
try:
    stats = requests.get(f"{BACKEND}/stats", timeout=2).json()
    total = stats.get("total", 0)
    attacks = stats.get("attacks", 0)
    status = "Operational"
except:
    total, attacks, status = 0, 0, "Offline"

col1, col2, col3 = st.columns(3)
with col1: st.metric("Ingested Traffic", f"{total:,}")
with col2: st.metric("Active Threats", f"{attacks:,}")
with col3: st.metric("System Status", status)

st.markdown("## Operational Parameters")
st.markdown("""
The Netwatch IDS is currently monitoring edge traffic through an ensemble-based heuristic engine. 
This interface provides high-fidelity visibility into sub-second network transactions.

*   **Heuristics**: Random Forest / Isolation Forest
*   **Explainability**: Per-event SHAP value derivation
*   **Persistence**: SQLite Real-time Logging
""")

st.divider()
st.caption("System Version: Developer Build v0.1.0")
