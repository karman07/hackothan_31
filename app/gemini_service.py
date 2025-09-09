import json
import re
import google.generativeai as genai

genai.configure(api_key="AIzaSyBvPqWVXQfZ4SDU-6VKUQt2QRjkqqQSIRo")
model = genai.GenerativeModel("gemini-1.5-flash")

def extract_certificate_details_with_gemini(extracted_text: str) -> dict:
    """
    Use Gemini to extract certificate details in strict JSON.
    """

    prompt = f"""
    Extract key details from the OCR text of a certificate.

    OCR Text:
    {extracted_text}

    Return STRICT JSON only. No explanation, no markdown, no code fences.
    JSON fields required:
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

    response = model.generate_content(prompt)

    cleaned_text = response.text.strip()

    # 🛠 Strip code fences if Gemini adds ```json ... ```
    if cleaned_text.startswith("```"):
        cleaned_text = re.sub(r"^```json|^```|```$", "", cleaned_text, flags=re.MULTILINE).strip()

    try:
        details = json.loads(cleaned_text)
    except Exception:
        details = {
            "candidate_name": None,
            "relation": None,
            "parent_name": None,
            "institute": None,
            "course": None,
            "division": None,
            "marks_obtained": None,
            "marks_total": None,
            "date": None,
            "place": None
        }

    return details
def check_certificate_authenticity(text: str, institute: str, image_bytes: bytes) -> str:
    response = model.generate_content([
        f"Extracted text: {text}\nInstitute: {institute}\n\nQuestion: Verify if this certificate looks authentic and belongs to {institute}.",
        {"mime_type": "image/png", "data": image_bytes}
    ])
    return response.text