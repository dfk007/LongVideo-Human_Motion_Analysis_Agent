from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.core.config import settings
import os

app = FastAPI(title=settings.PROJECT_NAME)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Motion Analysis Agent Backend is Running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "project": settings.GOOGLE_CLOUD_PROJECT,
        "model": settings.GEMINI_MODEL_NAME
    }