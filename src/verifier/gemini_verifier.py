import os
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

def verify_event(image_paths):
    parts = [{
        "text": """
You are verifying a classroom violation.

Allowed violations:
1. Student fight (physical aggression)
2. Student using a mobile phone

Ignore normal gestures.
If unsure, respond with "none".

Return ONLY JSON:
{
  "violation": "fight | phone | none",
  "confidence": 0-1,
  "reason": "short"
}
"""
    }]

    for img_path in image_paths:
        with open(img_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode()
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": img_base64
                }
            })

    payload = {"contents": [{"parts": parts}]}

    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro-vision:generateContent?key={API_KEY}"
    response = requests.post(url, json=payload)

    return response.json()
