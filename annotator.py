import os
import json
import time
import fitz  # PyMuPDF for extracting text from PDFs
import google.generativeai as genai  # Gemini API

# Configuration
DOWNLOAD_DIR = "NeurIPS_Papers"
METADATA_FILE = "metadata.json"
GEMINI_API_KEY = "AIzaSyB3oxGLI6-tS91k1LOl_llLIumo8AGjtnA"
LABELS = ["Reinforcement Learning", "Computer Vision", "Natural Language Processing", "Optimization", "Theoretical ML"]
API_RETRY_LIMIT = 5  # Maximum retries for API calls
API_RETRY_DELAY = 2  # Initial delay (seconds) for retrying

genai.configure(api_key=GEMINI_API_KEY)

def load_metadata():
    """Load metadata from JSON file."""
    metadata_path = os.path.join(DOWNLOAD_DIR, METADATA_FILE)
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)
    return []

def save_metadata(metadata):
    """Save metadata back to JSON file after adding annotations."""
    metadata_path = os.path.join(DOWNLOAD_DIR, METADATA_FILE)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

def is_valid_pdf(file_path):
    """Check if the PDF file exists and is not empty."""
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0

def extract_text_from_pdf(pdf_path):
    """Extract text from the first few pages of a PDF."""
    if not is_valid_pdf(pdf_path):
        return ""
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([doc[i].get_text("text") for i in range(min(3, len(doc)))]).strip()
        return text if text else "No text found"
    except Exception:
        return ""

def call_gemini_with_retry(prompt, max_retries=API_RETRY_LIMIT):
    """Retry Gemini API call if rate limit error occurs."""
    model = genai.GenerativeModel("gemini-pro")
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip() if response else "Unknown"
        except Exception as e:
            if "429" in str(e):
                wait_time = API_RETRY_DELAY * (2 ** attempt)
                time.sleep(wait_time)
            else:
                return "Unknown"
    return "Unknown"

def annotate_text(text):
    """Send extracted text to Gemini API for annotation."""
    prompt = f"""
    Categorize the following research paper text into one of these categories: {', '.join(LABELS)}.
    Respond with only the label name.
    Text:\n{text[:1000]}
    """
    return call_gemini_with_retry(prompt)

def annotate_papers():
    """Iterate over metadata, extract text, annotate, and update JSON."""
    metadata = load_metadata()
    if not metadata:
        return
    unannotated = [entry for entry in metadata if "annotation" not in entry]
    if not unannotated:
        return
    for entry in unannotated:
        pdf_path = entry.get("file_path")
        if not is_valid_pdf(pdf_path):
            entry["annotation"] = "Unknown"
            continue
        text = extract_text_from_pdf(pdf_path)
        if not text:
            entry["annotation"] = "Unknown"
            continue
        annotation = annotate_text(text)
        entry["annotation"] = annotation
        time.sleep(2)  # Delay to prevent hitting API rate limits
    save_metadata(metadata)

if __name__ == "__main__":
    start_time = time.time()
    annotate_papers()
    print(f"Total annotation time: {time.time() - start_time:.2f} seconds")