from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="Motion Analysis Agent API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Motion Analysis Agent Backend is Running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "project": os.getenv("GOOGLE_CLOUD_PROJECT"),
        "model": os.getenv("GEMINI_MODEL_NAME")
    }
