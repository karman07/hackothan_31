# main.py
import os
import json
import re
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import google.generativeai as genai
import easyocr
from bson import json_util

# ----------------- Load environment variables -----------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "test")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ----------------- MongoDB Setup -----------------
client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
users_collection = db["users"]

# ----------------- Gemini Setup -----------------
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# ----------------- EasyOCR Setup -----------------
reader = easyocr.Reader(["en"])

# ----------------- FastAPI Setup -----------------
app = FastAPI(title="Certificate OCR API")

# Enable CORS for all origins (temporary for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Helper Functions -----------------
def extract_text_from_image(image_bytes: bytes) -> str:
    results = reader.readtext(image_bytes, detail=0)
    return " ".join(results)

def extract_institute_name(extracted_text: str) -> str:
    lines = extracted_text.splitlines()
    for line in lines:
        if "institute" in line.lower():
            return line.strip()
    # Fallback: pick first line if nothing matches
    return lines[0].strip() if lines else "Unknown Institute"

def simple_fallback_parser(extracted_text: str) -> dict:
    """
    Simple regex-based parser to avoid null fields when Gemini fails.
    """
    candidate_match = re.search(r"certificate that[_\s]+([A-Z][a-z]+)", extracted_text, re.I)
    parent_match = re.search(r"(Son|Daughter|Wife) of Shri[_\s]+([A-Z][a-z]+)", extracted_text, re.I)
    marks_match = re.search(r"(\d+)[-_ ]*marks out of[_ ]*(\d+)", extracted_text, re.I)
    date_match = re.search(r"DATE\s*[:\-]\s*([0-9A-Za-z\-]+)", extracted_text, re.I)
    place_match = re.search(r"PLACE\s*[:\-]\s*([A-Za-z ]+)", extracted_text, re.I)
    course_match = re.search(r"Diploma[- ]?in[ ]([A-Za-z &]+)", extracted_text, re.I)
    division_match = re.search(r"(First|Second|Third)[ ]Division", extracted_text, re.I)

    return {
        "candidate_name": candidate_match.group(1) if candidate_match else "Unknown",
        "relation": parent_match.group(1) if parent_match else "Son/Daughter",
        "parent_name": parent_match.group(2) if parent_match else "Unknown",
        "institute": extract_institute_name(extracted_text),
        "course": course_match.group(1) if course_match else "Unknown Course",
        "division": division_match.group(1) if division_match else "Unknown",
        "marks_obtained": marks_match.group(1) if marks_match else "0",
        "marks_total": marks_match.group(2) if marks_match else "0",
        "date": date_match.group(1) if date_match else "Unknown Date",
        "place": place_match.group(1) if place_match else "Unknown Place",
        "exists_in_db": False
    }

async def extract_certificate_details_with_gemini(extracted_text: str) -> dict:
    """
    Extract structured certificate details using Gemini with fallback parsing.
    """
    prompt = f"""
    Extract key details from the OCR text of a certificate.
    OCR Text:
    {extracted_text}
    Return STRICT JSON only. JSON fields required:
    {{
      "candidate_name": string or null,
      "relation": string or null,
      "parent_name": string or null,
      "institute": string or null,
      "course": string or null,
      "division": string or null,
      "marks_obtained": string or null,
      "marks_total": string or null,
      "date": string or null,
      "place": string or null
    }}
    """
    try:
        response = model.generate_content(prompt)
        cleaned_text = response.text.strip()
        if cleaned_text.startswith("```"):
            cleaned_text = re.sub(r"^```json|^```|```$", "", cleaned_text, flags=re.MULTILINE).strip()
        details = json.loads(cleaned_text)
        # Replace any nulls with fallback parsing
        for k, v in details.items():
            if v is None or v == "":
                details[k] = simple_fallback_parser(extracted_text).get(k)
    except Exception:
        details = simple_fallback_parser(extracted_text)

    # MongoDB check
    candidate_name = details.get("candidate_name")
    parent_name = details.get("parent_name")
    if candidate_name and parent_name:
        query = {
            "candidate_name": {"$regex": candidate_name, "$options": "i"},
            "parent_name": {"$regex": parent_name, "$options": "i"}
        }
        user = await users_collection.find_one(query)
        details["exists_in_db"] = user is not None
    else:
        details["exists_in_db"] = False

    return details

async def check_certificate_authenticity(text: str, institute: str, image_bytes: bytes, key_details: dict) -> str:
    """
    RAG-style check: pass DB context to Gemini for authenticity verification.
    """
    candidate_name = key_details.get("candidate_name") or ""
    query = {"candidate_name": {"$regex": candidate_name, "$options": "i"}}
    user_records = await users_collection.find(query).to_list(length=5)
    context = f"Extracted DB Records:\n{json_util.dumps(user_records, indent=2)}"

    try:
        response = model.generate_content([
            f"Certificate OCR text: {text}\nInstitute: {institute}\n\n{context}\n\nQuestion: Verify if this certificate looks authentic and belongs to the candidate.",
            {"mime_type": "image/png", "data": image_bytes}
        ])
        return response.text
    except Exception:
        return "Gemini API quota exceeded or error occurred. Certificate details extracted locally."

# ----------------- API Endpoint -----------------
@app.post("/verify-certificate")
async def verify_certificate(file: UploadFile = File(...)):
    image_bytes = await file.read()
    extracted_text = extract_text_from_image(image_bytes)
    institute = extract_institute_name(extracted_text)
    key_details = await extract_certificate_details_with_gemini(extracted_text)
    authenticity = await check_certificate_authenticity(extracted_text, institute, image_bytes, key_details)

    return {
        "extracted_text": extracted_text,
        "institute": institute,
        "key_details": key_details,
        "authenticity_check": authenticity
    }
