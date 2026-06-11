import os
import json
import re
import uuid
import traceback
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# PDF & DOCX parsing
import fitz  # PyMuPDF
from docx import Document

# ML-based similarity scoring
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Google Gemini API
from groq import Groq

# ─── Config ──────────

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {"pdf", "docx"}


# Load environment variables from .env
load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# In-memory session store (use Redis/DB for production)
session_store = {}

# ─── Helpers ─────────────────

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(filepath):
    """Extract plain text from PDF using PyMuPDF."""
    text = ""
    with fitz.open(filepath) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()


def extract_text_from_docx(filepath):
    """Extract plain text from DOCX using python-docx."""
    doc = Document(filepath)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


def extract_resume_text(filepath):
    """Route to correct extractor based on file extension."""
    ext = filepath.rsplit(".", 1)[1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(filepath)
    elif ext == "docx":
        return extract_text_from_docx(filepath)
    return ""


def parse_resume_with_ai(raw_text):
    """Use Gemini API to extract structured data from resume text."""
    prompt = f"""You are a professional resume parser. Extract structured information from the resume below.

Resume Text:
{raw_text[:4000]}

Return ONLY a valid JSON object with these fields (use empty string "" or empty array [] if not found):
{{
  "name": "",
  "email": "",
  "phone": "",
  "location": "",
  "title": "",
  "summary": "",
  "totalExperience": "",
  "skills": [],
  "topSkills": [],
  "education": [
    {{"degree": "", "institution": "", "year": ""}}
  ],
  "experience": [
    {{"role": "", "company": "", "duration": "", "description": ""}}
  ],
  "certifications": [],
  "languages": [],
  "strengths": [],
  "projects": []
}}

Rules:
- topSkills: pick the 8 most relevant technical/professional skills
- strengths: 3 key professional strengths based on the resume
- projects: list project names/descriptions if mentioned
Return only JSON, no markdown backticks, no explanation."""

    message = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    max_tokens=1500,
    messages=[{"role": "user", "content": prompt}]
    )
    raw = message.choices[0].message.content.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    return json.loads(raw)


def compute_tfidf_similarity(resume_text, jd_text):
    """
    ML-based similarity using TF-IDF + Cosine Similarity (scikit-learn).
    Returns a score between 0 and 100.
    """
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(float(similarity[0][0]) * 100, 1)


def match_resume_with_jd(parsed_resume, raw_text, jd_text):
    """Use Gemini API + TF-IDF to match resume against job description."""

    # ML scoring
    tfidf_score = compute_tfidf_similarity(raw_text, jd_text)

    prompt = f"""You are an expert resume-job-description matching system.

Resume Summary:
- Name: {parsed_resume.get('name', '')}
- Title: {parsed_resume.get('title', '')}
- Skills: {', '.join(parsed_resume.get('skills', []))}
- Experience: {parsed_resume.get('totalExperience', '')}
- Summary: {parsed_resume.get('summary', '')}
- Education: {json.dumps(parsed_resume.get('education', []))}
- Projects: {json.dumps(parsed_resume.get('projects', []))}

Job Description:
{jd_text[:2500]}

TF-IDF Similarity Score (ML baseline): {tfidf_score}/100

Analyze the match and return ONLY valid JSON:
{{
  "aiMatchScore": 78,
  "matchLevel": "Strong Match",
  "matchedSkills": ["skill1", "skill2"],
  "missingSkills": ["skill3", "skill4"],
  "keyInsights": [
    {{"type": "positive", "text": "..."}},
    {{"type": "positive", "text": "..."}},
    {{"type": "warning", "text": "..."}},
    {{"type": "negative", "text": "..."}}
  ],
  "experienceMatch": 75,
  "skillsMatch": 82,
  "educationMatch": 70,
  "overallFit": "One concise sentence about the candidate's fit."
}}

matchLevel options: "Excellent Match" (>=85), "Strong Match" (70-84), "Good Match" (55-69), "Partial Match" (40-54), "Low Match" (<40)
aiMatchScore: Your AI assessment score 0-100 (consider the TF-IDF baseline but use your judgment)
Return only JSON, no markdown."""

    message = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    max_tokens=1500,
    messages=[{"role": "user", "content": prompt}]
    )
    raw = message.choices[0].message.content.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    result = json.loads(raw)
    result["tfidfScore"] = tfidf_score
    # Weighted final score: 70% AI + 30% TF-IDF
    result["finalScore"] = round(result["aiMatchScore"] * 0.7 + tfidf_score * 0.3, 1)
    return result


def generate_feedback(parsed_resume, match_result=None):
    """Use Gemini API to generate actionable resume improvement suggestions."""
    missing = []
    if match_result:
        missing = match_result.get("missingSkills", [])
        score = match_result.get("finalScore", "N/A")
    else:
        score = "N/A"

    prompt = f"""You are a professional career coach and resume expert. 
Analyze this resume and provide 6 specific, actionable improvement suggestions.

Resume:
- Name: {parsed_resume.get('name', '')}
- Title: {parsed_resume.get('title', '')}
- Skills: {', '.join(parsed_resume.get('skills', []))}
- Experience: {parsed_resume.get('totalExperience', '')}
- Summary: {parsed_resume.get('summary', '')}
- Education: {json.dumps(parsed_resume.get('education', []))}
{"- Job match score: " + str(score) + "/100" if score != "N/A" else ""}
{"- Missing skills for target role: " + ", ".join(missing) if missing else ""}

Return ONLY a valid JSON array of 6 items:
[
  {{
    "category": "Category Name",
    "priority": "high",
    "suggestion": "Specific, actionable suggestion with concrete steps.",
    "impact": "What improving this will achieve."
  }}
]

Categories: Summary, Skills, Experience, ATS Optimization, Keywords, Achievements, Formatting, Projects
Priority: "high", "medium", or "low"
Return only JSON array, no markdown."""

    message = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    max_tokens=1500,
    messages=[{"role": "user", "content": prompt}]
    )
    raw = message.choices[0].message.content.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    return json.loads(raw)


# ─── Routes ──────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_resume():
    """Upload and parse a resume (PDF or DOCX)."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file. Only PDF and DOCX allowed."}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(filepath)

    try:
        raw_text = extract_resume_text(filepath)
        if not raw_text.strip():
            return jsonify({"error": "Could not extract text from file. Try a different PDF."}), 400

        parsed = parse_resume_with_ai(raw_text)
        resume_id = uuid.uuid4().hex

        session_store[resume_id] = {
            "id": resume_id,
            "filename": filename,
            "filepath": filepath,
            "raw_text": raw_text,
            "parsed": parsed,
            "match_result": None,
            "feedback": None
        }

        return jsonify({
            "success": True,
            "resume_id": resume_id,
            "filename": filename,
            "parsed": parsed
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Parsing failed: {str(e)}"}), 500


@app.route("/api/match", methods=["POST"])
def match_resumes():
    """Match all uploaded resumes against a job description."""
    data = request.get_json()
    jd_text = data.get("jd_text", "").strip()
    resume_ids = data.get("resume_ids", [])

    if not jd_text:
        return jsonify({"error": "Job description is required"}), 400
    if not resume_ids:
        return jsonify({"error": "No resumes selected"}), 400

    results = []
    for rid in resume_ids:
        resume = session_store.get(rid)
        if not resume:
            continue
        try:
            match = match_resume_with_jd(
                resume["parsed"],
                resume["raw_text"],
                jd_text
            )
            match["resume_id"] = rid
            match["name"] = resume["parsed"].get("name", resume["filename"])
            match["filename"] = resume["filename"]
            session_store[rid]["match_result"] = match
            results.append(match)
        except Exception as e:
            traceback.print_exc()
            results.append({
                "resume_id": rid,
                "name": resume["parsed"].get("name", resume["filename"]),
                "filename": resume["filename"],
                "error": str(e),
                "finalScore": 0,
                "aiMatchScore": 0,
                "tfidfScore": 0,
                "matchLevel": "Error",
                "matchedSkills": [],
                "missingSkills": [],
                "keyInsights": [],
                "overallFit": "Analysis failed."
            })

    # Sort by finalScore descending
    results.sort(key=lambda x: x.get("finalScore", 0), reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return jsonify({"success": True, "results": results})


@app.route("/api/feedback", methods=["POST"])
def get_feedback():
    """Generate improvement feedback for a resume."""
    data = request.get_json()
    resume_id = data.get("resume_id")

    resume = session_store.get(resume_id)
    if not resume:
        return jsonify({"error": "Resume not found"}), 404

    try:
        feedback = generate_feedback(
            resume["parsed"],
            resume.get("match_result")
        )
        session_store[resume_id]["feedback"] = feedback
        return jsonify({"success": True, "feedback": feedback})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/resumes", methods=["GET"])
def list_resumes():
    """Return all uploaded resumes in the session."""
    resumes = []
    for rid, data in session_store.items():
        resumes.append({
            "id": rid,
            "filename": data["filename"],
            "name": data["parsed"].get("name", data["filename"])
        })
    return jsonify({"resumes": resumes})


@app.route("/api/clear", methods=["POST"])
def clear_session():
    """Clear all uploaded resumes."""
    for data in session_store.values():
        try:
            if os.path.exists(data["filepath"]):
                os.remove(data["filepath"])
        except:
            pass
    session_store.clear()
    return jsonify({"success": True})


# ─── Run ──────────────

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True, port=5000)