"""
Main Application Entry Point

This is the FastAPI application that serves the Motion Analysis Agent API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.api.routes import router
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
    Motion Analysis Agent API
    
    An agentic system for analyzing human motion in videos using:
    - Ollama Cloud for natural language reasoning
    - MediaPipe for human pose estimation
    - ChromaDB for vector-based evidence retrieval
    - FastAPI for the backend API layer
    
    Upload a video and ask questions like:
    - "Is this squat safe according to coaching standards?"
    - "Analyze the takeoff technique in this vault."
    - "What's the knee angle during the deepest part of the squat?"
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
# In production, replace with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex="https?://.*",  # Allows all origins (http/https)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api", tags=["Motion Analysis"])

# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Runs when the application starts.
    
    Logs configuration and validates setup.
    """
    logger.info("=" * 60)
    logger.info(f"{settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info("=" * 60)
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"LLM: Ollama Cloud ({settings.OLLAMA_MODEL})")
    logger.info(f"ChromaDB: {settings.CHROMA_URL}")
    logger.info(f"Upload Directory: {settings.UPLOAD_DIR}")
    logger.info(f"Processed Directory: {settings.PROCESSED_DIR}")
    logger.info("=" * 60)
    
    # Validate Ollama configuration
    if settings.OLLAMA_API_KEY:
        logger.info("✓ Ollama Cloud API Key configured")
    else:
        logger.warning("⚠️  Using local Ollama (no API key set)")
    
    logger.info("Application startup complete")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Runs when the application shuts down"""
    logger.info("Application shutting down...")

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """API root endpoint"""
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "upload": "POST /api/upload",
            "query": "POST /api/query",
            "health": "GET /api/health"
        }
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=settings.ENVIRONMENT == "development"
    )
