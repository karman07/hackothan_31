# filename: main.py
import pytesseract
import requests
import json
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from PIL import Image, UnidentifiedImageError
from io import BytesIO

# ------------------- CONFIG -------------------
API_KEY = "AIzaSyDDuCc_V3eZavSm91--KyZcjaPToF_MCPU"
GEMINI_API = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

app = FastAPI(title="OCR + Gemini Extractor")

# ✅ Enable CORS (accept all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Accept all origins
    allow_credentials=True,
    allow_methods=["*"],        # Allow all HTTP methods
    allow_headers=["*"],        # Allow all headers
)

# ------------------- OCR -------------------
def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract text from PDFs and all common image formats."""
    try:
        if filename.lower().endswith(".pdf"):
            images = convert_from_bytes(file_bytes)
            text = ""
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"
            return text.strip()
        else:  # jpg, png, jpeg, webp, tiff etc.
            image = Image.open(BytesIO(file_bytes))
            return pytesseract.image_to_string(image).strip()
    except UnidentifiedImageError:
        return ""
    except Exception as e:
        return f"Error reading file: {str(e)}"

# ------------------- Gemini Extract -------------------
def extract_with_gemini(text: str) -> dict:
    """Send OCR text to Gemini and enforce strict JSON output."""
    prompt = f"""
You are an information extraction assistant. 
From the given document text, extract the following details and return ONLY valid JSON.
Do not include explanations or extra text. Keys must always exist.

Required JSON fields:
{{
  "candidate_name": "...",
  "relation": "...",
  "parent_name": "...",
  "institute": "...",
  "course": "...",
  "division": "...",
  "marks_obtained": "...",
  "marks_total": "...",
  "date": "...",
  "place": "..."
}}

Document text:
{text}
"""

    body = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(GEMINI_API, headers=headers, data=json.dumps(body))

    if response.status_code != 200:
        return {"error": f"Gemini API error {response.status_code}", "details": response.text}

    try:
        extracted = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        # Ensure it's valid JSON
        return json.loads(extracted)
    except Exception as e:
        return {"error": "Failed to parse Gemini response", "details": str(e), "raw": response.text}

# ------------------- FastAPI Endpoint -------------------
@app.post("/extract-details/")
async def extract_details(file: UploadFile = File(...)):
    file_bytes = await file.read()
    extracted_text = extract_text(file_bytes, file.filename)

    if not extracted_text:
        return {"error": "No text could be extracted from the file."}

    details = extract_with_gemini(extracted_text)
    return {"extracted_text": extracted_text, "details": details}
