import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import apply_saas_theme, dashboard_header

# ── Setup ──
BACKEND = "http://localhost:8000"
st.set_page_config(page_title="Analytics", layout="wide")
apply_saas_theme()

dashboard_header("Security Analytics", "Incident Distribution & Temporal Trends")

try:
    stats = requests.get(f"{BACKEND}/stats", timeout=3).json()
    alerts = requests.get(f"{BACKEND}/alerts?limit=1000", timeout=3).json()
except Exception as e:
    st.error(f"Engine Unreachable: {e}")
    st.stop()

by_type = stats.get("by_type", {})

if not by_type:
    st.info("Data Pending / Insufficient Telemetry")
    st.stop()

# ── Trend Analysis (Plotly) ──
st.subheader("Temporal Trend")

if alerts:
    df = pd.DataFrame(alerts)
    from datetime import datetime
    local_tz = datetime.now().astimezone().tzinfo
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(local_tz).dt.tz_localize(None)
    df = df[df["attack_type"] != "Normal"]
    
    if not df.empty:
        df.set_index("timestamp", inplace=True)
        # Resample - 1T (Minute)
        minute_trends = df.resample("1min").size().reset_index(name="count")
        
        fig = px.area(minute_trends, x="timestamp", y="count", 
                     color_discrete_sequence=["#0ea5e9"]) # Light blue
        fig.update_layout(
            xaxis=dict(showgrid=False, title=""),
            yaxis=dict(showgrid=False, title="Incidents/Min"),
            margin=dict(l=0, r=0, t=20, b=0),
            height=250
        )
        st.plotly_chart(fig, width="stretch", theme="streamlit", config={'displayModeBar': False})
    else:
        st.caption("Monitoring Active / No Trends Captured")

# ── Distribution Analysis (Plotly) ──
st.divider()
st.subheader("Classification Distribution")

dist_df = pd.DataFrame(list(by_type.items()), columns=["Classification", "Count"]).sort_values("Count", ascending=True)

fig2 = px.bar(dist_df, x="Count", y="Classification", orientation="h",
             color_discrete_sequence=["#0ea5e9"])
fig2.update_layout(
    xaxis=dict(showgrid=False, title="Quantity"),
    yaxis=dict(showgrid=False, title=""),
    margin=dict(l=0, r=0, t=10, b=0),
    height=max(150, min(400, len(dist_df)*40 + 50))
)
st.plotly_chart(fig2, width="stretch", theme="streamlit", config={'displayModeBar': False})
