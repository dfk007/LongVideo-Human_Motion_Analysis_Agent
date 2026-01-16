"""
API Routes

This module defines the REST API endpoints for the motion analysis system.

Main endpoints:
- POST /upload: Upload and process a video
- POST /query: Ask questions about the video
- GET /health: Health check
"""

from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging

from app.services.video_service import VideoService
from app.services.workflow import workflow
from app.services.agent_service import agent_service

logger = logging.getLogger(__name__)

router = APIRouter()
video_service = VideoService()


# Request/Response Models

class QueryRequest(BaseModel):
    """Request model for video queries"""
    query: str
    exercise_type: Optional[str] = None  # Optional hint about exercise type


class Evidence(BaseModel):
    """Evidence citation in the response"""
    tool: str
    input: str
    observation: str


class QueryResponse(BaseModel):
    """Response model for queries with evidence grounding"""
    answer: str
    evidence: List[Dict]
    query: str


# API Endpoints

@router.post("/upload")
def upload_video(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...)
):
    """
    Upload and process a video.
    
    This endpoint:
    1. Validates the file format
    2. Saves the file to disk
    3. Triggers background processing (chunking, pose extraction, indexing)
    
    The background processing implements the "efficient long-video handling"
    requirement by chunking and indexing rather than loading everything into memory.
    
    Args:
        file: Video file (MP4, MOV, or AVI)
        
    Returns:
        Upload confirmation with filename
    """
    # Validate file format
    valid_extensions = ('.mp4', '.mov', '.avi')
    if not file.filename.lower().endswith(valid_extensions):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file format. Supported: {valid_extensions}"
        )
    
    logger.info(f"Received upload: {file.filename}")
    
    # Save file
    try:
        content = file.file.read()
        file_path = video_service.save_upload_sync(content, file.filename)
        logger.info(f"Saved file to: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to save file: {str(e)}"
        )
    
    # Trigger background processing
    # This runs asynchronously so the user doesn't wait for processing
    background_tasks.add_task(workflow.process_video, file_path, file.filename)
    
    return {
        "message": "Video uploaded successfully. Processing started in background.",
        "filename": file.filename,
        "note": "You can start querying in about 1-2 minutes per minute of video."
    }


@router.post("/query", response_model=QueryResponse)
def query_video(request: QueryRequest):
    """
    Answer a question about the uploaded video.
    
    This endpoint implements the agentic system:
    1. Receives natural language query
    2. Agent decides which tools to use (search, analyze, validate)
    3. Agent executes tools in sequence
    4. Agent generates evidence-grounded answer
    
    The response includes both the answer and the evidence trail,
    implementing the "grounding answers in evidence" requirement.
    
    Args:
        request: Query request with question and optional exercise type
        
    Returns:
        Answer with evidence citations (timestamps, measurements, tool calls)
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    logger.info(f"Processing query: {request.query}")
    
    try:
        # Use the agent to answer the query
        # The agent will:
        # 1. Search for relevant segments (retrieval)
        # 2. Extract pose data (perception)
        # 3. Validate against standards (reasoning)
        # 4. Generate grounded answer
        result = agent_service.answer_query(request.query)
        
        # Format evidence for response
        # Each piece of evidence shows which tool was used and what it found
        formatted_evidence = []
        for ev in result.get('evidence', []):
            formatted_evidence.append({
                "tool_used": ev.get('tool', 'unknown'),
                "tool_input": ev.get('input', ''),
                "finding": ev.get('observation', '')[:300]  # Truncate long observations
            })
        
        logger.info(f"Generated answer with {len(formatted_evidence)} evidence items")
        
        return QueryResponse(
            answer=result.get('answer', 'Unable to generate answer'),
            evidence=formatted_evidence,
            query=request.query
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns system status and configuration.
    """
    from app.core.config import settings
    
    return {
        "status": "healthy",
        "service": "Motion Analysis Agent",
        "model": settings.OLLAMA_MODEL,
        "features": {
            "pose_estimation": "MediaPipe",
            "agent_framework": "LangChain",
            "vector_db": "ChromaDB",
            "llm": "Ollama Cloud"
        }
    }


@router.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Motion Analysis Agent API",
        "version": "1.0",
        "endpoints": {
            "POST /api/upload": "Upload video for analysis",
            "POST /api/query": "Ask questions about uploaded video",
            "GET /api/health": "Health check"
        }
    }
