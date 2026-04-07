import streamlit as st
import requests
import io
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from utils import apply_saas_theme, dashboard_header, format_local_time

# ── Setup ──
BACKEND = "http://localhost:8000"
st.set_page_config(page_title="Audit / Export", layout="wide")
apply_saas_theme()

dashboard_header("Compliance Reports", "Automated Security Audit & Persistence Summaries")
st.markdown("Download a high-fidelity PDF summary of recent sessions, threat actor activity, and metric benchmarks.")

# Internal Helper for PDF
def generate_pdf(stats, alerts, ips):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # 1. Header (Minimalist)
    story.append(Paragraph("Netwatch IDS Security Audit", styles["Title"]))
    story.append(Paragraph(f"Timestamp (Local): {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 20))

    # 2. Executive Metrics
    story.append(Paragraph("Executive Summary", styles["Heading2"]))
    summary_data = [
        ["Metric", "Value"],
        ["Total Traffic Ingested", str(stats.get("total", 0))],
        ["Identified Attacks", str(stats.get("attacks", 0))],
        ["Alert Ratio", f"{(stats.get('attacks', 0) / max(stats.get('total', 1), 1) * 100):.4f}%"],
    ]
    t = Table(summary_data, colWidths=[200, 200])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.black),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
    ]))
    story.append(t)
    story.append(Spacer(1, 20))

    # 3. Recent Alerts
    story.append(Paragraph("Incident Logs", styles["Heading2"]))
    if alerts:
        alert_data = [["Time", "Source", "Type", "Severity"]] + [
            [format_local_time(a["timestamp"]), a["source_ip"], a["attack_type"].title(), a["severity"].title()]
            for a in alerts[:50]
        ]
        t2 = Table(alert_data, colWidths=[100, 120, 110, 100])
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.black),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        story.append(t2)

    doc.build(story)
    buffer.seek(0)
    return buffer

# UI Action Button
st.markdown("<br>", unsafe_allow_html=True)

def style_severity(val):
    v = str(val).upper()
    if v == "CRITICAL": return "color: #DC2626; font-weight: 600;" # Red
    if v == "HIGH": return "color: #EA580C; font-weight: 600;"    # Orange
    if v == "MEDIUM": return "color: #CA8A04;"                    # Yellow
    if v == "LOW": return "color: #6B7280;"                       # Gray
    return "color: #16A34A;"                                      # Green

st.subheader("Audit Preview")
try:
    recent_alerts = requests.get(f"{BACKEND}/alerts?limit=5", timeout=2).json()
    if recent_alerts:
        prev_df = pd.DataFrame(recent_alerts)[["timestamp", "source_ip", "attack_type", "severity"]]
        prev_df["timestamp"] = prev_df["timestamp"].apply(format_local_time)
        prev_df.columns = [c.title().replace("_", " ") for c in prev_df.columns]
        st.dataframe(
            prev_df.style.map(style_severity, subset=['Severity']),
            use_container_width=True,
            hide_index=True
        )
except:
    st.caption("Preview Unavailable: System Offline")

st.markdown("<br>", unsafe_allow_html=True)
if st.button("Generate Audit Report"):
    try:
        with st.spinner("Querying Database and Generating PDF..."):
            stats = requests.get(f"{BACKEND}/stats", timeout=3).json()
            alerts = requests.get(f"{BACKEND}/alerts?limit=100", timeout=3).json()
            ips = requests.get(f"{BACKEND}/ip-reputation", timeout=3).json()
            
            pdf_buffer = generate_pdf(stats, alerts, ips)
            
            st.success("Audit PDF securely synthesized.")
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name=f"netwatch_audit_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
    except Exception as e:
        st.error(f"Audit Failure: {e}")
