"""
SIF-Guard FastAPI backend.

Run locally:
    pip install fastapi uvicorn scikit-learn pandas numpy
    uvicorn main:app --reload --port 8000

Then in the dashboard's Header -> "Backend API Connection" field, set:
    http://localhost:8000/api/v1/analyze
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from inference import analyze_report

app = FastAPI(title="SIF-Guard ML API", version="1.0.0")

# CORS: allow the Vite dev server (default port 3000/5173) and any origin during development.
# Tighten this to your actual deployed frontend origin before going to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    report_text: str


class BatchAnalyzeRequest(BaseModel):
    report_texts: List[str]


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "service": "sif-guard-ml", "gemini_enabled": False}


@app.post("/api/v1/analyze")
def analyze(payload: AnalyzeRequest):
    if not payload.report_text or not payload.report_text.strip():
        raise HTTPException(status_code=400, detail="report_text must not be empty")
    try:
        return analyze_report(payload.report_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/v1/analyze-batch")
def analyze_batch(payload: BatchAnalyzeRequest):
    results = []
    for text in payload.report_texts:
        if not text or not text.strip():
            continue
        try:
            results.append(analyze_report(text))
        except Exception as e:
            results.append({"error": str(e), "report_text": text})
    return {"results": results}
