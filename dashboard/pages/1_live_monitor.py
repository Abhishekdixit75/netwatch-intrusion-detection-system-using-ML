import streamlit as st
import requests
import time
import pandas as pd
import plotly.express as px
from utils import apply_saas_theme, dashboard_header, format_local_time

# ── Setup ──
BACKEND = "http://localhost:8000"
st.set_page_config(page_title="Monitor / Live", layout="wide")
apply_saas_theme()

dashboard_header("Live Monitoring", "Network Traffic Heuristics / Real-time")

# Session state for the traffic pulse chart
if "traffic_history" not in st.session_state:
    st.session_state.traffic_history = pd.DataFrame(columns=["timestamp", "count"])

# ── Controller (Sidebar) ──
with st.sidebar:
    st.markdown("### Controls")
    is_live = st.toggle("Live Feed", value=True, help="Enable/Disable real-time data updates.")
    refresh_sec = st.select_slider("Refresh Interval (sec)", options=[1, 2, 5, 10, 30], value=2)
    st.markdown("---")
    st.markdown("### Session")
    st.caption("Node: Edge 01")

# ── Silent Live Fragment ──
@st.fragment(run_every=refresh_sec if is_live else None)
def monitor_fragment():
    try:
        # Fetch telemetry
        alerts_resp = requests.get(f"{BACKEND}/alerts?limit=100", timeout=3)
        stats_resp = requests.get(f"{BACKEND}/stats", timeout=3)

        alerts = alerts_resp.json() if alerts_resp.status_code == 200 else []
        stats = stats_resp.json() if stats_resp.status_code == 200 else {}

        # 1. System Telemetry Bar
        total = stats.get("total", 0)
        attacks = stats.get("attacks", 0)
        rate = (attacks / max(total, 1)) * 100

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Traffic Ingested", f"{total:,}")
        with c2: st.metric("Threats Detected", f"{attacks:,}")
        with c3: st.metric("Alert Ratio", f"{rate:.4f}%")
        with c4: st.metric("System Status", "Healthy")
        st.divider()

        # 2. Traffic Pulse Chart
        # Update history
        new_row = pd.DataFrame([{"timestamp": pd.Timestamp.now(), "count": total}])
        st.session_state.traffic_history = pd.concat([st.session_state.traffic_history, new_row]).tail(20)
        
        # Calculate delta for the pulse
        pulse_df = st.session_state.traffic_history.copy()
        pulse_df["delta"] = pulse_df["count"].diff().fillna(0)
        
        fig = px.area(pulse_df, x="timestamp", y="delta", color_discrete_sequence=["#0ea5e9"]) # Light blue
        fig.update_layout(
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(showgrid=False, visible=False),
            margin=dict(l=0, r=0, t=0, b=0),
            height=60,
        )
        st.plotly_chart(fig, use_container_width=True, theme="streamlit", config={'displayModeBar': False})
        st.caption("Packets per Second")

        # 3. Live Incident Log
        if not alerts:
            st.caption("No incidents captured in current session.")
        else:
            rows = []
            for a in alerts:
                rows.append({
                    "Incident ID": f"NET-{a['id']}",
                    "Timestamp": format_local_time(a["timestamp"]),
                    "Source Address": a["source_ip"],
                    "Classification": a["attack_type"].title(),
                    "Severity": a["severity"].title()
                })
            
            def style_severity(val):
                v = str(val).upper()
                if v == "CRITICAL": return "color: #DC2626; font-weight: 600;" # Red
                if v == "HIGH": return "color: #EA580C; font-weight: 600;"    # Orange
                if v == "MEDIUM": return "color: #CA8A04;"                    # Yellow
                if v == "LOW": return "color: #6B7280;"                       # Gray
                return "color: #16A34A;"                                      # Green

            df = pd.DataFrame(rows)
            st.dataframe(
                df.style.map(style_severity, subset=['Severity']), 
                use_container_width=True, 
                hide_index=True
            )

    except Exception as e:
        st.warning(f"Connection Interrupted: Reconnecting... ({e})")

# Execute the live component
monitor_fragment()
