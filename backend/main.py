"""
AEGIS AI - FastAPI Backend Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router

app = FastAPI(
    title="AEGIS AI — Ethical AI Auditing Framework",
    description="Detect, quantify, and explain bias in AI hiring systems.",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {
        "project": "AEGIS AI",
        "description": "Ethical AI Auditing Framework for Hiring System Bias",
        "problem_statement": "PS9 — Jedi Code Compliance System",
        "team": "MISAL PAV",
        "hackathon": "Bluebit 2026",
        "docs": "/docs"
    }
