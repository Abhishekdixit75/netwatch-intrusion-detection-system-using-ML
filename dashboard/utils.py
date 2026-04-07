import streamlit as st

def apply_saas_theme():
    """Injects a minimalist Enterprise dashboard design system, natively supporting Light/Dark modes."""
    st.markdown("""
        <style>
            /* 1. Reset & Typography */
            @import url('https://rsms.me/inter/inter.css');
            html, body, [data-testid="stAppViewContainer"] {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }
            
            /* 2. Streamlit Overrides */
            #MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; display: none; }
            [data-testid="stToolbar"] { right: 2rem; top: 1rem; }
            
            /* 3. Layout */
            .block-container { padding: 3rem 5rem !important; }
            
            /* 4. Enterprise Headers */
            h1 { font-size: 1.5rem !important; font-weight: 600; letter-spacing: -0.02em; margin-bottom: 0.5rem; }
            h2 { font-size: 1.1rem !important; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem; }
            h3 { font-size: 0.9rem !important; font-weight: 600; color: var(--text-color) !important; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.05em; }

            /* 5. Metrics */
            div[data-testid="stMetric"] {
                background-color: var(--secondary-background-color);
                padding: 1.5rem !important;
                border-radius: 4px;
                border: 1px solid var(--secondary-background-color);
            }
            [data-testid="stMetricValue"] { font-size: 1.75rem !important; font-weight: 500 !important; color: var(--text-color); }
            [data-testid="stMetricLabel"] { font-size: 0.75rem !important; color: var(--text-color) !important; opacity: 0.8; font-weight: 500; }

            /* 6. Clean Borders & Tables */
            .stDataFrame, div[data-testid="stTable"], .stTable { 
                border-radius: 4px !important; 
            }
            
            /* 7. Plotly Chart Baseline */
            .js-plotly-plot .plotly .modebar { display: none !important; }
            
        </style>
    """, unsafe_allow_html=True)

def dashboard_header(title: str, subtitle: str = None):
    """Clean enterprise header component."""
    st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p style='color: var(--text-color); opacity: 0.7; font-size:0.95rem; margin-top:-0.5rem;'>{subtitle}</p>", unsafe_allow_html=True)
    st.markdown("<div style='height:1px; background: var(--secondary-background-color); margin: 1.5rem 0 2rem 0;'></div>", unsafe_allow_html=True)

def format_local_time(iso_str: str, format_str: str = "%H:%M:%S") -> str:
    import pandas as pd
    from datetime import datetime
    try:
        local_tz = datetime.now().astimezone().tzinfo
        # Ensure string is evaluated as UTC to avoid naive local parsing
        dt = pd.to_datetime(iso_str, utc=True)
        return dt.tz_convert(local_tz).strftime(format_str)
    except:
        return iso_str[11:19]
