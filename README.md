# 🌾 AI-Powered Crop Recommendation System
### Powered by IBM watsonx.ai AutoAI + Python Flask

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![IBM watsonx.ai](https://img.shields.io/badge/IBM-watsonx.ai-blue)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple)

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [IBM watsonx.ai Setup](#ibm-watsonxai-setup)
5. [Local Development](#local-development)
6. [APP_CONFIG Reference](#app_config-reference)
7. [Deploying to IBM Code Engine](#deploying-to-ibm-code-engine)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)

---

## Overview

This application uses an **IBM watsonx.ai AutoAI** classification model trained on the [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) to recommend the most suitable crop based on:

| Parameter | Unit | Range |
|---|---|---|
| Nitrogen (N) | kg/ha | 0 – 200 |
| Phosphorus (P) | kg/ha | 0 – 200 |
| Potassium (K) | kg/ha | 0 – 250 |
| Temperature | °C | -10 – 60 |
| Humidity | % | 0 – 100 |
| Soil pH | — | 0 – 14 |
| Rainfall | mm | 0 – 3000 |

---

## Project Structure

```
crop-recommendation-app/
├── app.py                   # Flask backend + APP_CONFIG
├── requirements.txt         # Python dependencies
├── .env.example             # Credentials template (copy → .env)
├── .env                     # ← YOUR credentials (never commit this)
├── .gitignore
├── Procfile                 # For Cloud Foundry / Heroku deployment
├── README.md
└── templates/
    └── index.html           # Full-stack frontend (Bootstrap 5)
```

---

## Prerequisites

- Python **3.9+**
- An **IBM Cloud** account → [cloud.ibm.com](https://cloud.ibm.com)
- An **IBM Watson Studio** project with an **AutoAI** experiment trained and deployed
- `pip` / a virtual environment tool

---

## IBM watsonx.ai Setup

### Step 1 — Create an IBM Cloud API Key
1. Log in to [cloud.ibm.com](https://cloud.ibm.com)
2. Navigate to **Manage → IAM → API Keys**
3. Click **Create** → name it `CropAI-Key` → copy the key immediately

### Step 2 — Train an AutoAI Experiment
1. Open **Watson Studio** → create or open a project
2. **Add asset → AutoAI experiment**
3. Upload the crop recommendation CSV (columns: `N, P, K, temperature, humidity, ph, rainfall, label`)
4. Set **prediction column** = `label`, prediction type = **Multiclass Classification**
5. Run the experiment and select the best pipeline

### Step 3 — Deploy the Model
1. In the AutoAI experiment results, click **Save as → Model**
2. Navigate to **Deployments → New deployment** → select the saved model
3. Choose **Online** deployment → click **Create**
4. Open the deployment → copy the **Endpoint URL** (scoring URL)

### Step 4 — Collect Your Credentials
You need three values:

```
IBM_API_KEY          = <your IAM API key from Step 1>
DEPLOYMENT_URL       = <scoring URL from Step 3, ends with /predictions?version=...>
WATSONX_PROJECT_ID   = <Project ID from project Settings page>
```

---

## Local Development

### 1. Clone / download the project
```bash
git clone https://github.com/your-org/crop-recommendation-app.git
cd crop-recommendation-app
```

### 2. Create a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure credentials
```bash
cp .env.example .env
```
Open `.env` and fill in your values:
```ini
IBM_API_KEY=your_actual_api_key
DEPLOYMENT_URL=https://us-south.ml.cloud.ibm.com/ml/v4/deployments/<id>/predictions?version=2021-05-01
WATSONX_PROJECT_ID=your_project_id
```

### 5. Run the app
```bash
python app.py
```
Open your browser at **http://localhost:5000**

To enable debug mode:
```bash
# Windows PowerShell
$env:FLASK_DEBUG="true"; python app.py

# macOS / Linux
FLASK_DEBUG=true python app.py
```

---

## APP_CONFIG Reference

Open `app.py` and find the `APP_CONFIG` block (≈ line 33). Every key is documented:

| Key | Default | Description |
|---|---|---|
| `IBM_API_KEY` | *(from .env)* | IBM Cloud IAM API key |
| `DEPLOYMENT_URL` | *(from .env)* | watsonx.ai AutoAI scoring endpoint |
| `IAM_TOKEN_URL` | `https://iam.cloud.ibm.com/identity/token` | IAM token URL (don't change unless non-standard region) |
| `WATSONX_PROJECT_ID` | *(from .env)* | Watson Studio project ID |
| `APP_TITLE` | `CropAI` | Displayed in header and browser tab |
| `APP_SUBTITLE` | `Intelligent Crop Recommendation System` | Hero subtitle |
| `APP_DESCRIPTION` | `Powered by IBM watsonx.ai AutoAI` | Badge text |
| `THEME_COLOR` | `#2e7d32` | Primary accent colour (CSS hex) |
| `THEME_COLOR_LIGHT` | `#66bb6a` | Lighter accent |
| `THEME_COLOR_DARK` | `#1b5e20` | Darker gradient colour |
| `DEBUG` | `false` | Flask debug mode |
| `PORT` | `5000` | HTTP port |
| `SECRET_KEY` | *(from .env or default)* | Flask session secret |

---

## Deploying to IBM Code Engine

### Option A – Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "app:app"]
```

```bash
# Build and push
docker build -t crop-recommendation-app .
docker tag crop-recommendation-app icr.io/<namespace>/crop-recommendation-app:latest
docker push icr.io/<namespace>/crop-recommendation-app:latest
```

Then deploy via the IBM Code Engine console or CLI:
```bash
ibmcloud ce application create \
  --name crop-recommendation-app \
  --image icr.io/<namespace>/crop-recommendation-app:latest \
  --env IBM_API_KEY=<key> \
  --env DEPLOYMENT_URL=<url> \
  --env WATSONX_PROJECT_ID=<id> \
  --port 8080
```

### Option B – Cloud Foundry / Heroku (Procfile)
```
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2
```
Set environment variables via your platform's dashboard or CLI.

---

## API Reference

### `GET /`
Returns the main application HTML page.

### `POST /predict`
**Request body (JSON):**
```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.8,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 202.9
}
```

**Success response (200):**
```json
{
  "success": true,
  "crop": {
    "name": "Rice",
    "emoji": "🌾",
    "season": "Kharif",
    "description": "Rice thrives in hot, humid climates with abundant water supply."
  },
  "inputs": { "N": 90, "P": 42, "K": 43, "temperature": 20.8,
               "humidity": 82.0, "ph": 6.5, "rainfall": 202.9 }
}
```

**Validation error (422):**
```json
{
  "success": false,
  "errors": { "ph": "ph must be between 0 and 14." }
}
```

**Server error (5xx):**
```json
{
  "success": false,
  "error": "Model API returned 401. Check your DEPLOYMENT_URL and IBM_API_KEY."
}
```

### `GET /health`
Returns `{"status":"ok","app":"CropAI","version":"1.0.0"}` — useful for load-balancer probes.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `IBM_API_KEY is not configured` | `.env` not loaded or key missing | Copy `.env.example` → `.env`, fill values |
| `Model API returned 401` | API key invalid or expired | Regenerate key in IBM Cloud IAM |
| `Model API returned 404` | Wrong `DEPLOYMENT_URL` | Re-copy the scoring URL from the deployment page |
| `Unexpected response structure` | AutoAI response schema changed | Check raw response in logs; adjust `call_watsonx_model()` |
| `Cannot connect to IBM watsonx.ai` | Network/firewall issue | Verify internet access and endpoint URL |
| App starts but page is blank | Port already in use | Change `PORT` in `.env` |

---

## .gitignore (recommended)

```
.env
.venv/
__pycache__/
*.pyc
*.pyo
.DS_Store
instance/
*.egg-info/
dist/
build/
```

---

*Built with ❤️ using IBM watsonx.ai AutoAI and Python Flask.*
