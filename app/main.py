# ----------------- Imports -----------------
import os
import json
import re
import cv2
import numpy as np
import qrcode
import hashlib
from skimage.metrics import structural_similarity as ssim
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import google.generativeai as genai
import easyocr
from bson import json_util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from datetime import datetime
from pyzbar.pyzbar import decode as decode_qr
from PIL import Image, ExifTags
import io

# ----------------- Load environment variables -----------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "test")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ----------------- MongoDB Setup -----------------
client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
users_collection = db["users"]
calls_collection = db["calls"]

# ----------------- Gemini Setup -----------------
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# ----------------- EasyOCR Setup -----------------
reader = easyocr.Reader(["en"])

# ----------------- FastAPI Setup -----------------
app = FastAPI(title="Certificate Verification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Helpers -----------------
def normalize_text(text: str) -> str:
    text = re.sub(r"[_\-\[\]{}():;@]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()

def fix_common_ocr_errors(text: str) -> str:
    corrections = {
        "dipfoma": "diploma",
        "cechnical": "technical",
        "nbustrial": "industrial",
        "mbato": "board"
    }
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    return text

def is_strict_name_match(name1: str, name2: str, threshold: float = 0.65) -> bool:
    if not name1 or not name2:
        return False
    n1, n2 = normalize_text(name1), normalize_text(name2)
    vectorizer = TfidfVectorizer().fit([n1, n2])
    tfidf_matrix = vectorizer.transform([n1, n2])
    cosine_sim = float(cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0])
    fuzzy_sim = fuzz.token_sort_ratio(n1, n2) / 100.0
    return ((cosine_sim * 0.5) + (fuzzy_sim * 0.5)) >= threshold

# ----------------- OCR + Parsing -----------------
def extract_text_from_image(image_bytes: bytes) -> str:
    results = reader.readtext(image_bytes, detail=0)
    return fix_common_ocr_errors(" ".join(results))

def extract_institute_name(extracted_text: str) -> str:
    lines = extracted_text.splitlines()
    for line in lines:
        if "institute" in line.lower() or "college" in line.lower():
            return line.strip()
    return lines[0].strip() if lines else "Unknown Institute"

# ----------------- Metadata Extraction -----------------
def extract_metadata(file_bytes: bytes) -> dict:
    metadata = {}
    try:
        image = Image.open(io.BytesIO(file_bytes))
        metadata["format"] = image.format
        metadata["mode"] = image.mode
        exif_data = image._getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                metadata[tag] = str(value)
    except Exception:
        metadata["error"] = "Cannot read metadata"
    return metadata

# ----------------- AI Detection Placeholder -----------------
def detect_ai_generated_content(text: str) -> bool:
    """Placeholder for AI-generated text detection."""
    return False

# ----------------- Forgery/Tamper Detection -----------------
def detect_ela_anomalies(image_bytes: bytes) -> bool:
    try:
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, enc_img = cv2.imencode(".jpg", img, encode_param)
        resaved = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)
        diff = cv2.absdiff(img, resaved)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        anomaly_ratio = np.sum(thresh > 0) / (thresh.shape[0] * thresh.shape[1])
        return anomaly_ratio > 0.02
    except Exception:
        return False

# ----------------- QR / Hash Validation -----------------
def extract_qr_and_verify(image_bytes: bytes, file_hash: str) -> dict:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    decoded_objs = decode_qr(img)
    if not decoded_objs:
        return {"qr_found": False, "qr_valid": None}  # ✅ None means "not applicable"
    try:
        qr_data = json.loads(decoded_objs[0].data.decode("utf-8"))
        qr_valid = qr_data.get("hash") == file_hash
        return {"qr_found": True, "qr_valid": qr_valid, "qr_data": qr_data}
    except Exception:
        return {"qr_found": True, "qr_valid": False}

def compute_file_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()

# ----------------- Gemini Extractor -----------------
async def extract_certificate_details_with_gemini(extracted_text: str) -> dict:
    prompt = f"""
    Extract key details from the OCR text of a certificate.
    OCR Text:
    {extracted_text}
    Return STRICT JSON only with fields:
    candidate_name, parent_name, institute, course, division, marks_obtained, marks_total, date, place, certificate_id
    """
    try:
        response = model.generate_content(prompt)
        cleaned = response.text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```json|^```|```$", "", cleaned, flags=re.MULTILINE).strip()
        details = json.loads(cleaned)
    except Exception:
        details = {}
    return details

# ----------------- Sanitize for Mongo -----------------
def sanitize_for_mongo(doc):
    if isinstance(doc, dict):
        return {k: sanitize_for_mongo(v) for k, v in doc.items()}
    elif isinstance(doc, list):
        return [sanitize_for_mongo(v) for v in doc]
    elif isinstance(doc, (np.bool_, np.int64, np.float64)):
        return doc.item()
    return doc

# ----------------- API Endpoint -----------------
@app.post("/verify-certificate")
async def verify_certificate(cert_file: UploadFile = File(...)):
    cert_bytes = await cert_file.read()
    extracted_text = extract_text_from_image(cert_bytes)
    institute = extract_institute_name(extracted_text)

    # Step 1: Extract details with Gemini
    key_details = await extract_certificate_details_with_gemini(extracted_text)

    # Step 2: DB Cross-check
    candidate_name = normalize_text(key_details.get("candidate_name", ""))
    exists = False
    if candidate_name:
        users = await users_collection.find({}).to_list(length=1000)
        for user in users:
            db_name = normalize_text(user.get("candidate_name") or "")
            if is_strict_name_match(candidate_name, db_name):
                exists = True
                break
    key_details["exists_in_db"] = exists

    # Step 3: QR / Hash validation
    file_hash = compute_file_hash(cert_bytes)
    qr_result = extract_qr_and_verify(cert_bytes, file_hash)

    # Step 4: Forgery detection
    tampered = detect_ela_anomalies(cert_bytes)

    # Step 5: Metadata + AI check
    metadata = extract_metadata(cert_bytes)
    ai_generated = detect_ai_generated_content(extracted_text)

    # ✅ Legitimacy Decision
    if qr_result["qr_found"] is False:
        is_legitimate = exists and not tampered
    else:
        is_legitimate = exists and qr_result.get("qr_valid", False) and not tampered

    # ✅ Save log in DB
    log_entry = {
        "timestamp": datetime.utcnow(),
        "candidate_name": key_details.get("candidate_name", "Unknown"),
        "institute": key_details.get("institute", institute),
        "course": key_details.get("course", "Unknown"),
        "hash": file_hash,
        "exists_in_db": exists,
        "qr_result": qr_result,
        "tampered": bool(tampered),
        "is_legitimate": bool(is_legitimate),
        "metadata": metadata,
        "ai_generated": ai_generated,
        "key_details": key_details,
    }
    await calls_collection.insert_one(sanitize_for_mongo(log_entry))

    # ✅ API Response
    return {
        "extracted_text": extracted_text,
        "institute": institute,
        "key_details": key_details,
        "db_match": exists,
        "qr_result": qr_result,
        "tampered": bool(tampered),
        "file_hash": file_hash,
        "is_legitimate": bool(is_legitimate),
        "metadata": metadata,
        "ai_generated": ai_generated
    }

# ----------------- Startup -----------------
@app.on_event("startup")
async def startup_event():
    users = await users_collection.find({}).to_list(length=20)
    print("📌 Users in DB at startup:")
    print(json_util.dumps(users, indent=2))
