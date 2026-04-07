import streamlit as st
import requests
import pandas as pd
from utils import apply_saas_theme, dashboard_header

# ── Setup ──
BACKEND = "http://localhost:8000"
st.set_page_config(page_title="Reputation / Intelligence", layout="wide")
apply_saas_theme()

dashboard_header("Source Intelligence", "Actor Reputation Tracking & Analysis")

# ── Controller ──
with st.sidebar:
    st.markdown("### Filters")
    view_blocked = st.checkbox("Show Blocked Only", value=False)
    st.markdown("---")
    st.markdown("### Session")
    st.caption("Node: Edge 01")

try:
    data = requests.get(f"{BACKEND}/ip-reputation", timeout=3).json()
except Exception as e:
    st.error(f"Engine Unreachable: {e}")
    st.stop()

if not data:
    st.info("Monitoring Active: No Actors Flagged")
    st.stop()

# ── Intelligence Summary ──
df = pd.DataFrame(data)

# Force status to lowercase for robust metric counting
df["status_low"] = df["status"].str.lower()

c1, c2, c3 = st.columns(3)
c1.metric("Actors Flagged", len(df))
c2.metric("Monitoring Watch", len(df[df["status_low"] == "watching"]))
c3.metric("Blocked Nodes", len(df[df["status_low"] == "blocked"]))
st.divider()

# ── Asset Reputation Matrix ──
st.subheader("Reputation Matrix")

# Professional technical labels & Filtering
if view_blocked:
    df_disp = df[df["status_low"] == "blocked"].copy()
else:
    df_disp = df.copy()

display_cols = ["ip", "alert_count", "first_seen", "last_seen", "status"]
df_disp = df_disp[display_cols].copy()
df_disp.columns = [c.title() for c in df_disp.columns] # Title case
df_disp.rename(columns={"Ip": "IP Address"}, inplace=True)
df_disp = df_disp.sort_values("Alert_Count", ascending=False)

def style_status(val):
    v = str(val).lower()
    if v == "blocked": return "color: #DC2626; font-weight: 600;" # Red
    if v == "watching": return "color: #0ea5e9;"                  # Blue
    return "color: #6B7280;"                                      # Gray

st.dataframe(
    df_disp.style.map(style_status, subset=['Status']), 
    width="stretch", 
    hide_index=True
)

st.caption("Reputation database synchronized in real-time.")
