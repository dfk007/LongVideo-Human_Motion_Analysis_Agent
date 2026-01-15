import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Motion Analysis Agent"
    GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "mycompanion-daudfarzand89-4jan")
    GOOGLE_CLOUD_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    # New Model Names
    VERTEX_AI_MODEL: str = os.getenv("VERTEX_AI_MODEL", "gemini-1.5-flash")
    VERTEX_AI_EMBEDDING_MODEL: str = os.getenv("VERTEX_AI_EMBEDDING_MODEL", "text-embedding-004")
    
    CHROMA_HOST: str = os.getenv("CHROMA_HOST", "chromadb")
    CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", 8000))
    
    UPLOAD_DIR: str = "/app/uploads"
    PROCESSED_DIR: str = "/app/processed"

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.PROCESSED_DIR, exist_ok=True)