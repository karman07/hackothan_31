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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# ----------------- Cosine Similarity Helper -----------------
def is_similar_name(name1: str, name2: str, threshold: float = 0.6) -> bool:
    """
    Compare two text fields using cosine similarity (TF-IDF).
    More lenient: default threshold ~0.6.
    Also allow substring match.
    """
    if not name1 or not name2:
        return False

    n1, n2 = name1.strip().lower(), name2.strip().lower()

    if n1 in n2 or n2 in n1:
        return True

    vectorizer = TfidfVectorizer().fit([n1, n2])
    tfidf_matrix = vectorizer.transform([n1, n2])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    return similarity >= threshold

# ----------------- Helper Functions -----------------
def extract_text_from_image(image_bytes: bytes) -> str:
    results = reader.readtext(image_bytes, detail=0)
    return " ".join(results)

def extract_institute_name(extracted_text: str) -> str:
    lines = extracted_text.splitlines()
    for line in lines:
        if "institute" in line.lower():
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

async def extract_certificate_details_with_gemini(extracted_text: str) -> dict:
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
        for k, v in details.items():
            if v is None or v == "":
                details[k] = simple_fallback_parser(extracted_text).get(k)
    except Exception:
        details = simple_fallback_parser(extracted_text)

    # ----------------- MongoDB Matching (course/place focused) -----------------
    candidate_name = details.get("candidate_name")
    parent_name = details.get("parent_name")
    course = details.get("course")
    place = details.get("place")

    exists = False
    if course or place:
        users = await users_collection.find({}).to_list(length=1000)
        for user in users:
            # Strong anchors
            course_match = is_similar_name(course, user.get("course", ""), threshold=0.5)
            place_match = is_similar_name(place, user.get("place", ""), threshold=0.5)

            # Names are secondary, more lenient
            cand_match = is_similar_name(candidate_name, user.get("candidate_name", ""), threshold=0.4)
            parent_match = is_similar_name(parent_name, user.get("parent_name", ""), threshold=0.4)

            # Match logic (prioritize course/place)
            if (course_match and place_match) or (cand_match and course_match) or (cand_match and place_match):
                print(f"✅ Matched with user in DB: {user.get('candidate_name')} | {user.get('course')} @ {user.get('place')}")
                exists = True
                break
    details["exists_in_db"] = exists
    return details

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

# ----------------- Startup Event -----------------
@app.on_event("startup")
async def startup_event():
    users = await users_collection.find({}).to_list(length=100)
    print("📌 All Users in DB at startup:")
    print(json_util.dumps(users, indent=2))
