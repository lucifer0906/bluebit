"""
AEGIS AI — Setup Configuration
Ethical AI Auditing Framework for Hiring Systems
Team: MISAL PAV | Bluebit Hackathon 2026
"""

from setuptools import setup, find_packages

setup(
    name="aegis-ai",
    version="1.0.0",
    description="Ethical AI Auditing Framework — Bias Detection in Hiring Systems",
    author="Team MISAL PAV",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "xgboost>=2.0",
        "shap>=0.43",
        "lime>=0.2",
        "plotly>=5.18",
        "matplotlib>=3.8",
        "seaborn>=0.13",
        "fastapi>=0.104",
        "uvicorn>=0.24",
        "streamlit>=1.29",
        "jinja2>=3.1",
        "joblib>=1.3",
        "pydantic>=2.5",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4",
            "pytest-cov>=4.1",
            "httpx>=0.25",
        ]
    },
)
