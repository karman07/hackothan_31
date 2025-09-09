import re

def extract_institute_name(text: str) -> str:
    patterns = [
        r"(Institute of [A-Za-z ]+)",
        r"(University of [A-Za-z ]+)",
        r"([A-Za-z ]+ College)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    return "Not found"
