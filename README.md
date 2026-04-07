# Netwatch IDS

Netwatch IDS is a network intrusion detection system (NIDS) designed for real-time traffic monitoring, utilizing an ensemble machine learning architecture combined with Explainable AI (XAI) for forensic attribution.

## Architecture Overview

The system operates across three decoupled layers: a traffic simulator, a high-throughput inference API, and an adaptive Security Operations Center (SOC) dashboard.

### 1. Telemetry & Simulation (`/simulator`)
- `csv_simulator.py`: Replays the UNSW-NB15 dataset as live network traffic. It supports targeted routing of high-severity attacks and persistent IP generation to test the backend's stateful reputation blocking.

### 2. Inference Engine (`/backend` & `/ml_pipeline`)
- `backend/main.py`: The core FastAPI application. It exposes endpoints for telemetry ingestion, historical analytics, and SHAP-based model comparison.
- `backend/predict.py`: Handles raw feature scaling and runs real-time inference using a persistent Random Forest classifier and an Isolation Forest anomaly detector. Explanations are derived per-packet using SHAP.
- `backend/ip_tracker.py` & `backend/severity.py`: Maintains a stateful table of external IP addresses. Calculates threat severity dynamically based on attack categorization and historical recurrence, issuing automatic blocks for repeat offenders.
- `ml_pipeline/train.py`: Standalone scripts for dataset preprocessing, label encoding, and cross-validated ensemble model training. 

### 3. Command Center (`/dashboard`)
- `dashboard/app.py`: The entry point for the Streamlit SOC frontend. 
- Features fragment-based silent refreshing for non-blocking UI updates.
- Modules include live traffic heartbeats, Plotly-based temporal analytics, individual incident XAI diagnostics, and automated PDF compliance reporting via ReportLab.

## Local Setup

### Prerequisites
- Python 3.12
- `uv` package manager

### Installation

1. Clone the repository and install dependencies using `uv`:
   ```bash
   git clone <repository_url>
   cd network-intrusion-detection
   uv sync
   ```

2. (Optional) Rebuild the ML models. If you are starting fresh, execute the training pipeline first:
   ```bash
   uv run python ml_pipeline/train.py
   uv run python ml_pipeline/evaluate.py
   ```

### Execution

The application requires three terminal instances to operate simultaneously. 

1. **Start the Inference API**
   ```bash
   uv run uvicorn backend.main:app
   ```

2. **Start the SOC Dashboard**
   ```bash
   uv run streamlit run dashboard/app.py
   ```

3. **Start the Traffic Simulator**  
   Configure the simulator to push data to the API. Use the targeted flag for demonstration purposes if needed.
   ```bash
   uv run python simulator/csv_simulator.py --interval 1 --loop --targeted --persistent-ips
   ```

## Design Notes

The frontend UI strictly implements Streamlit's native semantic CSS layout parameters, enabling zero-configuration adaptive Light and Dark modes. All metrics and matrices automatically invert based on the host operating system's color scheme preferences.
