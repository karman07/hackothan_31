import os
import json
import re
import cv2
import numpy as np
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

# ----------------- Load environment variables -----------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "test")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ----------------- MongoDB Setup -----------------
client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
users_collection = db["users"]
calls_collection = db["calls"]   # ✅ new collection for verification logs

# ----------------- Gemini Setup -----------------
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# ----------------- EasyOCR Setup -----------------
reader = easyocr.Reader(["en"])

# ----------------- FastAPI Setup -----------------
app = FastAPI(title="Certificate OCR + Signature Verification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Helpers -----------------
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[_\-\[\]{}():;@]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fix_common_ocr_errors(text: str) -> str:
    corrections = {
        "daughtert": "daughter",
        "fart-tic": "part-time",
        "prpfvmatrort": "professional",
        "dipfoma": "diploma",
        "dcertificate-conrse": "certificate course",
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
    final_score = (cosine_sim * 0.5) + (fuzzy_sim * 0.5)
    return final_score >= threshold

# ----------------- Image Similarity -----------------
def load_image_from_bytes(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def compare_images_orb(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = orb.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
    if des1 is None or des2 is None:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if not matches:
        return 0.0
    good_matches = [m for m in matches if m.distance < 50]
    return float(len(good_matches) / max(len(matches), 1))

def compare_images_ssim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h, w = min(gray1.shape[0], gray2.shape[0]), min(gray1.shape[1], gray2.shape[1])
    gray1 = cv2.resize(gray1, (w, h))
    gray2 = cv2.resize(gray2, (w, h))
    score, _ = ssim(gray1, gray2, full=True)
    return float(score)

def compare_signature_watermark(sample_bytes: bytes, cert_bytes: bytes) -> float:
    img1 = load_image_from_bytes(sample_bytes)
    img2 = load_image_from_bytes(cert_bytes)
    orb_score = compare_images_orb(img1, img2)
    ssim_score = compare_images_ssim(img1, img2)
    return float((orb_score * 0.6) + (ssim_score * 0.4))

# ----------------- OCR + Parsing -----------------
def extract_text_from_image(image_bytes: bytes) -> str:
    results = reader.readtext(image_bytes, detail=0)
    text = " ".join(results)
    text = fix_common_ocr_errors(text)
    return text

def extract_institute_name(extracted_text: str) -> str:
    lines = extracted_text.splitlines()
    for line in lines:
        if "institute" in line.lower() or "college" in line.lower():
            return line.strip()
    return lines[0].strip() if lines else "Unknown Institute"

def simple_fallback_parser(extracted_text: str) -> dict:
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

# ----------------- Gemini Extractor -----------------
async def extract_certificate_details_with_gemini(extracted_text: str) -> dict:
    prompt = f"""
    Extract key details from the OCR text of a certificate.
    OCR Text:
    {extracted_text}
    Return STRICT JSON only.
    """
    try:
        response = model.generate_content(prompt)
        cleaned_text = response.text.strip()
        if cleaned_text.startswith("```"):
            cleaned_text = re.sub(r"^```json|^```|```$", "", cleaned_text, flags=re.MULTILINE).strip()
        details = json.loads(cleaned_text)
        for k, v in details.items():
            if v is None or v == "":
                details[k] = simple_fallback_parser(extracted_text).get(k)
    except Exception:
        details = simple_fallback_parser(extracted_text)

    candidate_name = details.get("candidate_name") or details.get("name") or ""
    candidate_name = normalize_text(candidate_name)

    exists = False
    if candidate_name:
        users = await users_collection.find({}).to_list(length=1000)
        for user in users:
            db_name = normalize_text(user.get("candidate_name") or "")
            if is_strict_name_match(candidate_name, db_name):
                exists = True
                break

    details["exists_in_db"] = exists
    return details

# ----------------- Authenticity Check -----------------
async def check_certificate_authenticity(text: str, institute: str, image_bytes: bytes, key_details: dict) -> str:
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
async def verify_certificate(sample_file: UploadFile = File(...), cert_file: UploadFile = File(...)):
    sample_bytes = await sample_file.read()
    cert_bytes = await cert_file.read()

    extracted_text = extract_text_from_image(cert_bytes)
    institute = extract_institute_name(extracted_text)
    key_details = await extract_certificate_details_with_gemini(extracted_text)
    authenticity = await check_certificate_authenticity(extracted_text, institute, cert_bytes, key_details)

    similarity_score = compare_signature_watermark(sample_bytes, cert_bytes)
    signature_match = similarity_score > 0.65
    is_legitimate = key_details.get("exists_in_db", False) and signature_match

    # ✅ Save verification call log in "calls" collection
    log_entry = {
        "timestamp": datetime.utcnow(),
        "candidate_name": key_details.get("candidate_name", "Unknown"),
        "institute": key_details.get("institute", "Unknown"),
        "course": key_details.get("course", "Unknown"),
        "signature_similarity_score": float(similarity_score),
        "signature_match": signature_match,
        "is_legitimate": is_legitimate,
        "key_details": key_details,
        "authenticity_check": authenticity
    }
    await calls_collection.insert_one(log_entry)

    return {
        "extracted_text": extracted_text,
        "institute": institute,
        "key_details": key_details,
        "authenticity_check": authenticity,
        "signature_similarity_score": float(similarity_score),
        "signature_match": signature_match,
        "is_legitimate": is_legitimate
    }

# ----------------- Startup Event -----------------
@app.on_event("startup")
async def startup_event():
    users = await users_collection.find({}).to_list(length=100)
    print("📌 All Users in DB at startup:")
    print(json_util.dumps(users, indent=2))
