# AI Resume Analyzer

An AI-powered resume analyzer built with **Python**, **Flask**, **Groq AI (LLaMA 3.3)**, and **scikit-learn** — built as part of an AI Engineer Intern assignment.

## Features

- **Resume Upload & Parsing** — supports PDF (PyMuPDF) and DOCX (python-docx)
- **Structured Data Extraction** — Name, Email, Phone, Skills, Education, Experience, Projects
- **Resume vs JD Matching** — hybrid scoring: 70% Groq AI + 30% TF-IDF cosine similarity (scikit-learn)
- **Match Score & Insights** — skill gap analysis, experience match, key insights
- **Resume Ranking System** — ranked leaderboard across multiple candidates
- **Feedback Suggestions** — AI-generated improvement tips with priority levels

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.x |
| Web Framework | Flask |
| AI / LLM | Groq API (LLaMA 3.3 70B) |
| ML Scoring | scikit-learn (TF-IDF + Cosine Similarity) |
| PDF Parsing | PyMuPDF (fitz) |
| DOCX Parsing | python-docx |
| Frontend | HTML5, CSS3, Vanilla JavaScript |

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-resume-analyzer.git
cd ai-resume-analyzer
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
python app.py
```

Open your browser at: **http://127.0.0.1:5000**

> **Note:** The API key is already configured in `app.py` and the app runs out of the box.

## How to Use

1. **Upload Resumes** — drag & drop or click to upload PDF/DOCX files
2. **View Parsed Data** — see extracted name, skills, experience, education
3. **Paste Job Description** — enter the target role's JD
4. **Analyze Match** — get AI + ML match scores with skill gaps
5. **Rankings** — compare multiple candidates on a leaderboard
6. **Feedback** — get prioritized resume improvement suggestions

## Project Structure

```
ai-resume-analyzer/
├── app.py              # Flask backend + all API routes
├── templates/
│   └── index.html      # Frontend UI
├── uploads/            # Temporary file storage (gitignored)
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main UI |
| POST | `/api/upload` | Upload & parse a resume |
| POST | `/api/match` | Match resumes against JD |
| POST | `/api/feedback` | Generate improvement feedback |
| GET | `/api/resumes` | List all uploaded resumes |
| POST | `/api/clear` | Clear session data |

## Scoring Methodology

The match score is calculated as a **weighted hybrid**:

```
Final Score = (AI Score × 0.7) + (TF-IDF Score × 0.3)
```

- **AI Score (70%)**: LLaMA 3.3 analyzes semantic fit, role alignment, skill relevance
- **TF-IDF Score (30%)**: scikit-learn cosine similarity on resume vs JD text