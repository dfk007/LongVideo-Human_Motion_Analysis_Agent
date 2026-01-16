"""
Application Configuration

This module centralizes all configuration settings for the motion analysis system.
Settings are loaded from environment variables with sensible defaults.
"""

import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    The configuration is organized into logical groups:
    - Project metadata
    - API keys and authentication
    - Model configuration
    - Database connections
    - Video processing parameters
    - MediaPipe settings
    - Storage paths
    """
    
    # Project Metadata
    PROJECT_NAME: str = "Motion Analysis Agent"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # API Keys
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    GOOGLE_CLOUD_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    # Model Configuration
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
    
    # Backend Server Configuration
    BACKEND_HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", 8000))
    
    # Database Configuration (PostgreSQL)
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "motion_analysis_db")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "db")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", 5432))
    
    @property
    def DATABASE_URL(self) -> str:
        """Construct PostgreSQL database URL"""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )
    
    # Vector Database Configuration (ChromaDB)
    CHROMA_HOST: str = os.getenv("CHROMA_HOST", "chromadb")
    CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", 8000))
    
    @property
    def CHROMA_URL(self) -> str:
        """Construct ChromaDB URL"""
        return f"http://{self.CHROMA_HOST}:{self.CHROMA_PORT}"
    
    # Video Processing Configuration
    CHUNK_DURATION: int = int(os.getenv("CHUNK_DURATION", 15))
    """Duration of each video chunk in seconds. 
    
    15 seconds is chosen because:
    - Most exercise reps take 3-8 seconds
    - Chunks capture 1-3 full reps
    - Small enough for efficient processing
    - Large enough to see complete movements
    """
    
    FRAME_SAMPLE_RATE: int = int(os.getenv("FRAME_SAMPLE_RATE", 3))
    """Process every Nth frame from video.
    
    With 30fps video and sample_rate=3, we process at 10fps.
    This is sufficient for human motion analysis (< 5 Hz movements).
    Reduces computational load by 3x while maintaining accuracy.
    """
    
    MAX_VIDEO_LENGTH: int = int(os.getenv("MAX_VIDEO_LENGTH", 3600))
    """Maximum video length in seconds (default: 1 hour)"""
    
    # MediaPipe Configuration
    MEDIAPIPE_MODEL_COMPLEXITY: int = int(os.getenv("MEDIAPIPE_MODEL_COMPLEXITY", 1))
    """MediaPipe model complexity: 0=Lite, 1=Full, 2=Heavy.
    
    1 (Full) is chosen as the best balance between:
    - Accuracy: Good enough for coaching feedback
    - Speed: Fast enough for real-time processing
    - Resource usage: Runs well on standard hardware
    """
    
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float = float(
        os.getenv("MEDIAPIPE_MIN_DETECTION_CONFIDENCE", 0.5)
    )
    """Minimum confidence for pose detection (0.0 - 1.0)"""
    
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE: float = float(
        os.getenv("MEDIAPIPE_MIN_TRACKING_CONFIDENCE", 0.5)
    )
    """Minimum confidence for pose tracking (0.0 - 1.0)"""
    
    # Storage Paths
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/app/uploads")
    """Directory for uploaded videos"""
    
    PROCESSED_DIR: str = os.getenv("PROCESSED_DIR", "/app/processed")
    """Directory for processed video chunks and pose data"""
    
    # LangChain Configuration (optional)
    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_API_KEY: Optional[str] = os.getenv("LANGCHAIN_API_KEY")
    
    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()

# Ensure required directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.PROCESSED_DIR, exist_ok=True)

# Validate critical settings
if not settings.GOOGLE_API_KEY:
    import warnings
    warnings.warn(
        "GOOGLE_API_KEY not set! The system will not work without it. "
        "Please set GOOGLE_API_KEY in your .env file."
    )
