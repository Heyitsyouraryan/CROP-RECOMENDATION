"""
AI-Powered Crop Recommendation System
Backend - Flask + IBM watsonx.ai AutoAI

APP_CONFIG
----------
Edit the APP_CONFIG dictionary below to customise:
  - IBM Cloud credentials / endpoints
  - Application display title and subtitle
  - UI theme accent colour (CSS hex)
  - Debug / port settings

All sensitive values fall back to environment variables (loaded
from the .env file via python-dotenv), so you can override any
APP_CONFIG entry without touching this file.
"""

import os
import json
import logging
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────
# 1.  Load .env (won't overwrite real env vars already set)
#     Falls back to .env.example when no .env file is present.
# ──────────────────────────────────────────────────────────────
_env_file = ".env" if os.path.exists(".env") else ".env.example"
load_dotenv(_env_file)

# ──────────────────────────────────────────────────────────────
# 2.  APP_CONFIG  ← easy-to-edit central configuration block
# ──────────────────────────────────────────────────────────────
APP_CONFIG = {
    # ── IBM Cloud / watsonx.ai credentials ──────────────────
    "IBM_API_KEY":       os.getenv("IBM_API_KEY",       "your_ibm_cloud_api_key_here"),
    "DEPLOYMENT_URL":    os.getenv("DEPLOYMENT_URL",    "your_watsonx_deployment_scoring_url_here"),
    "IAM_TOKEN_URL":     os.getenv("IAM_TOKEN_URL",     "https://iam.cloud.ibm.com/identity/token"),
    "WATSONX_PROJECT_ID": os.getenv("WATSONX_PROJECT_ID", "your_project_or_space_id_here"),

    # ── Application display settings ────────────────────────
    "APP_TITLE":         "CropAI",
    "APP_SUBTITLE":      "Intelligent Crop Recommendation System",
    "APP_DESCRIPTION":   "Powered by IBM watsonx.ai AutoAI",
    "THEME_COLOR":       "#2e7d32",   # deep green — change to any CSS hex
    "THEME_COLOR_LIGHT": "#66bb6a",   # lighter accent
    "THEME_COLOR_DARK":  "#1b5e20",   # darker shade for gradients

    # ── Flask settings ───────────────────────────────────────
    "DEBUG":    os.getenv("FLASK_DEBUG", "false").lower() == "true",
    "PORT":     int(os.getenv("PORT", 5000)),
    "SECRET_KEY": os.getenv("SECRET_KEY", "cropai-secret-key-change-in-production"),
}

# ──────────────────────────────────────────────────────────────
# 3.  Flask application
# ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = APP_CONFIG["SECRET_KEY"]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 4.  IBM Cloud helpers
# ──────────────────────────────────────────────────────────────
def get_iam_token() -> str:
    """Exchange the IBM Cloud API key for a short-lived IAM Bearer token."""
    api_key = APP_CONFIG["IBM_API_KEY"]
    if not api_key or api_key.startswith("your_"):
        raise ValueError("IBM_API_KEY is not configured. "
                         "Set it in .env or APP_CONFIG.")

    response = requests.post(
        APP_CONFIG["IAM_TOKEN_URL"],
        data={
            "grant_type":  "urn:ibm:params:oauth:grant-type:apikey",
            "apikey":      api_key,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def call_watsonx_model(features: list) -> str:
    """
    Send a single feature vector to the watsonx.ai AutoAI scoring endpoint
    and return the predicted crop label.

    The AutoAI REST payload format:
    {
        "input_data": [{
            "fields": ["N","P","K","temperature","humidity","ph","rainfall"],
            "values": [[<N>, <P>, <K>, <temp>, <humidity>, <ph>, <rainfall>]]
        }]
    }
    """
    deployment_url = APP_CONFIG["DEPLOYMENT_URL"]
    if not deployment_url or deployment_url.startswith("your_"):
        raise ValueError("DEPLOYMENT_URL is not configured. "
                         "Set it in .env or APP_CONFIG.")

    token   = get_iam_token()
    payload = {
        "input_data": [{
            "fields": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
            "values": [features],
        }]
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }

    logger.info("Calling watsonx.ai endpoint …")
    response = requests.post(deployment_url, json=payload,
                             headers=headers, timeout=60)
    response.raise_for_status()

    result = response.json()
    logger.info("Raw watsonx response: %s", json.dumps(result, indent=2))

    # Locate the predictions block — AutoAI uses either key.
    predictions_block = result.get("predictions") or result.get("results")
    if not predictions_block:
        raise ValueError(f"Unexpected response structure (no predictions key): {result}")

    block  = predictions_block[0]
    fields = block.get("fields", [])
    values = block.get("values", [[]])[0]   # first (and only) row

    # Build a field→value lookup so we are not sensitive to column order.
    row = dict(zip(fields, values))
    logger.info("Parsed prediction row: %s", row)

    # AutoAI deployments expose the label under one of these field names.
    LABEL_FIELDS = ["predicted_label", "prediction", "predictedLabel",
                    "Predicted_label", "label", "class"]

    for key in LABEL_FIELDS:
        if key in row and not isinstance(row[key], list):
            return str(row[key])

    # If none of the known label fields matched, find the first non-list,
    # non-numeric scalar in the row (skips probability arrays and scores).
    for key, val in row.items():
        if not isinstance(val, (list, float, int)):
            return str(val)

    # Last resort: if the whole row is just a probability vector, pick the
    # class with the highest probability using the CROP_INFO order.
    for key, val in row.items():
        if isinstance(val, list) and all(isinstance(v, (int, float)) for v in val):
            # Match index to the ordered crop list from CROP_INFO
            crop_list = list(CROP_INFO.keys())
            if len(val) == len(crop_list):
                best_idx = val.index(max(val))
                return crop_list[best_idx]
            # Generic fallback: return the index of the max probability
            best_idx = val.index(max(val))
            return f"class_{best_idx}"

    raise ValueError(f"Could not extract a crop label from response row: {row}")


# ──────────────────────────────────────────────────────────────
# 5.  Crop meta-data (icons, descriptions, season info)
# ──────────────────────────────────────────────────────────────
CROP_INFO = {
    "rice":        {"emoji": "🌾", "season": "Kharif",  "description": "Rice thrives in hot, humid climates with abundant water supply."},
    "maize":       {"emoji": "🌽", "season": "Kharif",  "description": "Maize grows well in warm temperatures with moderate rainfall."},
    "chickpea":    {"emoji": "🫘", "season": "Rabi",    "description": "Chickpea is a cool-season legume that fixes atmospheric nitrogen."},
    "kidneybeans": {"emoji": "🫘", "season": "Kharif",  "description": "Kidney beans prefer warm weather and well-drained fertile soil."},
    "pigeonpeas":  {"emoji": "🫘", "season": "Kharif",  "description": "Pigeon peas are drought-tolerant and improve soil fertility."},
    "mothbeans":   {"emoji": "🫘", "season": "Kharif",  "description": "Moth beans are highly drought-resistant pulse crops."},
    "mungbean":    {"emoji": "🫘", "season": "Kharif",  "description": "Mung bean is a short-duration, heat-tolerant legume."},
    "blackgram":   {"emoji": "🫘", "season": "Kharif",  "description": "Black gram is rich in protein and fixes soil nitrogen."},
    "lentil":      {"emoji": "🫘", "season": "Rabi",    "description": "Lentils prefer cool temperatures and well-drained soils."},
    "pomegranate": {"emoji": "🍎", "season": "Annual",  "description": "Pomegranate is drought-tolerant and thrives in semi-arid regions."},
    "banana":      {"emoji": "🍌", "season": "Annual",  "description": "Banana requires a warm, humid climate with rich, moist soil."},
    "mango":       {"emoji": "🥭", "season": "Summer",  "description": "Mango flourishes in tropical climates with a distinct dry season."},
    "grapes":      {"emoji": "🍇", "season": "Rabi",    "description": "Grapes require a warm, dry climate with cool nights."},
    "watermelon":  {"emoji": "🍉", "season": "Summer",  "description": "Watermelon loves heat, sandy soil, and long sunny days."},
    "muskmelon":   {"emoji": "🍈", "season": "Summer",  "description": "Muskmelon prefers sandy loam soil and hot, dry weather."},
    "apple":       {"emoji": "🍎", "season": "Rabi",    "description": "Apple requires a cool climate with cold winters for dormancy."},
    "orange":      {"emoji": "🍊", "season": "Rabi",    "description": "Oranges grow best in subtropical regions with mild winters."},
    "papaya":      {"emoji": "🍈", "season": "Annual",  "description": "Papaya grows quickly in tropical climates with well-drained soil."},
    "coconut":     {"emoji": "🥥", "season": "Annual",  "description": "Coconut palms thrive in humid coastal tropical regions."},
    "cotton":      {"emoji": "🌿", "season": "Kharif",  "description": "Cotton needs warm temperatures, deep black soil, and dry harvest conditions."},
    "jute":        {"emoji": "🌿", "season": "Kharif",  "description": "Jute is a rain-loving crop grown in hot, humid climates."},
    "coffee":      {"emoji": "☕", "season": "Annual",  "description": "Coffee grows in tropical highlands with moderate rainfall and shade."},
}


def get_crop_info(crop_name: str) -> dict:
    """Return display metadata for a predicted crop, with a safe fallback."""
    key = crop_name.lower().replace(" ", "")
    info = CROP_INFO.get(key, {
        "emoji":       "🌱",
        "season":      "Varies",
        "description": f"{crop_name.title()} is the recommended crop for your soil and climate conditions.",
    })
    return {**info, "name": crop_name.title()}


# ──────────────────────────────────────────────────────────────
# 6.  Routes
# ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Render the main application page."""
    return render_template("index.html", config=APP_CONFIG)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept JSON with soil/weather parameters, call watsonx.ai,
    and return a JSON prediction result.
    """
    try:
        data = request.get_json(force=True)
        required_fields = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        errors = {}

        parsed = {}
        for field in required_fields:
            raw = data.get(field)
            if raw is None or str(raw).strip() == "":
                errors[field] = f"{field} is required."
                continue
            try:
                value = float(raw)
            except (ValueError, TypeError):
                errors[field] = f"{field} must be a number."
                continue
            parsed[field] = value

        # ── Field-level range validation ─────────────────────
        ranges = {
            "N":           (0, 200),
            "P":           (0, 200),
            "K":           (0, 250),
            "temperature": (-10, 60),
            "humidity":    (0, 100),
            "ph":          (0, 14),
            "rainfall":    (0, 3000),
        }
        for field, (lo, hi) in ranges.items():
            if field in parsed and not (lo <= parsed[field] <= hi):
                errors[field] = f"{field} must be between {lo} and {hi}."

        if errors:
            return jsonify({"success": False, "errors": errors}), 422

        features = [parsed[f] for f in required_fields]
        crop     = call_watsonx_model(features)
        info     = get_crop_info(crop)

        logger.info("Prediction: %s  |  inputs: %s", crop, parsed)
        return jsonify({"success": True, "crop": info, "inputs": parsed})

    except ValueError as exc:
        logger.warning("Configuration error: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 503

    except requests.HTTPError as exc:
        logger.error("watsonx.ai HTTP error: %s", exc)
        return jsonify({"success": False,
                        "error": f"Model API returned {exc.response.status_code}. "
                                 "Check your DEPLOYMENT_URL and IBM_API_KEY."}), 502

    except requests.ConnectionError:
        logger.error("Cannot reach watsonx.ai endpoint.")
        return jsonify({"success": False,
                        "error": "Cannot connect to IBM watsonx.ai. "
                                 "Check your internet connection and DEPLOYMENT_URL."}), 503

    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during prediction.")
        return jsonify({"success": False,
                        "error": f"Unexpected server error: {exc}"}), 500


@app.route("/health")
def health():
    """Simple health-check endpoint for load-balancers / deployment probes."""
    return jsonify({
        "status":  "ok",
        "app":     APP_CONFIG["APP_TITLE"],
        "version": "1.0.0",
    })


# ──────────────────────────────────────────────────────────────
# 7.  Entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=APP_CONFIG["PORT"],
        debug=APP_CONFIG["DEBUG"],
    )
