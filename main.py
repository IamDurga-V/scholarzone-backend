from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore

if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

# Groq
from groq import Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

from scholarship_matcher import ScholarshipMatcher
from scholarship_scraper import run_all_scrapers

app = FastAPI(title="ScholarZone Backend API", version="1.0.0")

# Allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

matcher = ScholarshipMatcher(db)

# ── Models ──────────────────────────────────────────────
class StudentProfile(BaseModel):
    fullName:        Optional[str] = ""
    category:        Optional[str] = "general"
    state:           Optional[str] = ""
    educationLevel:  Optional[str] = ""
    annualIncome:    Optional[str] = "500000"
    percentage:      Optional[str] = "60"
    gender:          Optional[str] = "female"
    disability:      Optional[str] = "no"
    course:          Optional[str] = ""
    yearOfStudy:     Optional[str] = ""

class MatchRequest(BaseModel):
    profile: StudentProfile
    top_k:   Optional[int] = 20

# ── Routes ──────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ScholarZone Backend is running ✅", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/match")
async def match_scholarships(req: MatchRequest):
    """
    Main AI matching endpoint.
    1. FAISS finds top candidates from Firestore scholarships
    2. Groq explains each match with checklist-style reasons
    """
    try:
        profile = req.profile
        top_k   = req.top_k

        # Step 1 — get all scholarships from Firestore
        snap = db.collection("scholarships").where("is_active", "==", True).stream()
        scholarships = [{"id": d.id, **d.to_dict()} for d in snap]

        if not scholarships:
            raise HTTPException(status_code=404, detail="No scholarships found. Run /scrape first.")

        # Step 2 — FAISS matching
        candidates = matcher.match(profile.dict(), scholarships, top_k=top_k)

        # Step 3 — Groq explains matches
        explained = await explain_with_groq(profile.dict(), candidates)

        return {"success": True, "matches": explained, "total": len(explained)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scholarships")
def get_scholarships(limit: int = 100, state: str = None, category: str = None):
    """Get all scholarships from Firestore with optional filters"""
    try:
        query = db.collection("scholarships").where("is_active", "==", True)
        snap  = query.stream()
        results = []
        for d in snap:
            data = {"id": d.id, **d.to_dict()}
            # Filter by state if provided
            if state and state != "all":
                states = data.get("eligibility", {}).get("states", ["all"])
                if "all" not in states and state not in states:
                    continue
            # Filter by category if provided
            if category and category != "all":
                cats = data.get("eligibility", {}).get("categories", ["all"])
                if "all" not in cats and category not in cats:
                    continue
            results.append(data)
        return {"success": True, "scholarships": results[:limit], "total": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scholarships/{scholarship_id}")
def get_scholarship(scholarship_id: str):
    """Get a single scholarship by ID"""
    try:
        doc = db.collection("scholarships").document(scholarship_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Scholarship not found")
        return {"success": True, "scholarship": {"id": doc.id, **doc.to_dict()}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scrape")
async def scrape_scholarships():
    """
    Trigger all scrapers to fetch fresh scholarship data
    and store in Firestore. Run this weekly.
    """
    try:
        count = run_all_scrapers(db)
        # Rebuild FAISS index after scraping
        snap  = db.collection("scholarships").where("is_active", "==", True).stream()
        scholarships = [{"id": d.id, **d.to_dict()} for d in snap]
        matcher.build_index(scholarships)
        return {"success": True, "message": f"Scraped and stored {count} scholarships", "count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scrape/status")
def scrape_status():
    """Check how many scholarships are in Firestore"""
    try:
        snap  = db.collection("scholarships").stream()
        count = sum(1 for _ in snap)
        return {"success": True, "total_scholarships": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/profile/analyze")
async def analyze_profile(profile: StudentProfile):
    """
    Groq analyzes profile completeness and gives strength rating + suggestions
    """
    try:
        prompt = f"""Analyze this student scholarship profile and give a strength rating.

Profile:
- Name: {profile.fullName}
- Category: {profile.category}
- State: {profile.state}
- Education: {profile.educationLevel}
- Course: {profile.course}
- Income: ₹{profile.annualIncome}
- Percentage: {profile.percentage}%
- Gender: {profile.gender}
- Disability: {profile.disability}

Return ONLY valid JSON, no markdown:
{{
  "strength": "Strong" | "Good" | "Weak",
  "score": 85,
  "summary": "One sentence summary of profile strength",
  "positives": ["reason1", "reason2", "reason3"],
  "suggestions": ["suggestion1", "suggestion2"],
  "eligible_estimate": 12
}}"""

        result = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        text  = result.choices[0].message.content
        clean = text.replace("```json", "").replace("```", "").strip()
        import json
        data  = json.loads(clean)
        return {"success": True, "analysis": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Helper: Groq explains matches ───────────────────────
async def explain_with_groq(profile: dict, candidates: list) -> list:
    if not candidates:
        return []

    prompt = f"""You are a scholarship matching assistant. Explain why each scholarship matches this student.

Student:
- Category: {profile.get('category')}
- State: {profile.get('state')}
- Income: ₹{profile.get('annualIncome')}
- Education: {profile.get('educationLevel')}
- Percentage: {profile.get('percentage')}%
- Gender: {profile.get('gender')}
- Disability: {profile.get('disability')}
- Course: {profile.get('course')}

Scholarships to explain:
{chr(10).join([f"ID:{s['id']}|{s.get('name','')}" for s in candidates[:20]])}

Return ONLY valid JSON array, no markdown:
[
  {{
    "id": "scholarship_id",
    "matchPercent": 94,
    "checks": [
      {{"label": "Category", "status": "pass", "reason": "SC category qualifies"}},
      {{"label": "Income", "status": "pass", "reason": "₹1.2L is below ₹2.5L limit"}},
      {{"label": "State", "status": "pass", "reason": "Tamil Nadu is eligible"}},
      {{"label": "Education", "status": "pass", "reason": "UG level matches"}},
      {{"label": "Marks", "status": "warning", "reason": "85% meets 75% minimum"}}
    ],
    "summary": "One line reason why this is a great match",
    "highlights": ["Key point 1", "Key point 2"]
  }}
]
Order by matchPercent descending. Only include matches above 40%."""

    result = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    text  = result.choices[0].message.content
    clean = text.replace("```json", "").replace("```", "").strip()

    import json
    parsed = json.loads(clean)

    # Merge Groq explanation with scholarship data
    id_map = {s["id"]: s for s in candidates}
    merged = []
    for item in parsed:
        s = id_map.get(item["id"])
        if s:
            merged.append({**s, **item})

    return merged


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)