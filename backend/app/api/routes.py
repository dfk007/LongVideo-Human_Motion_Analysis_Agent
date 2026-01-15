from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from app.services.video_service import VideoService
from app.services.workflow import workflow
from app.services.vector_db import vector_db
from app.services.gemini_service import gemini_service
from pydantic import BaseModel
from typing import List

router = APIRouter()
video_service = VideoService()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    evidence: List[dict]

@router.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(('.mp4', '.mov', '.avi')):
        raise HTTPException(status_code=400, detail="Invalid file format")
        
    # Save file
    try:
        content = await file.read()
        file_path = await video_service.save_upload(content, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
    # Trigger processing in background
    background_tasks.add_task(workflow.process_video, file_path, file.filename)
    
    return {"message": "Video uploaded and processing started", "filename": file.filename}

@router.post("/query", response_model=QueryResponse)
async def query_video(request: QueryRequest):
    # 1. Search Vector DB for relevant segments
    results = vector_db.search_similar(request.query, n_results=3)
    
    # Extract contexts
    contexts = []
    evidence = []
    
    if results and results['documents']:
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            contexts.append(f"Timestamp {meta['start_time']}s - {meta['end_time']}s: {doc}")
            evidence.append(meta)
            
    context_str = "\n".join(contexts)
    
    # 2. Ask Gemini to reason about the query given the context
    # Note: For a "Production" system, we might re-watch the specific clips here 
    # if the description isn't enough. For now, we reason on the descriptions (RAG).
    
    prompt = f"""
    You are an expert human motion analysis agent.
    User Question: "{request.query}"    
    Here is evidence from the video analysis (timestamps and descriptions):
    {context_str}
    
    Based ONLY on this evidence, answer the user's question. 
    Cite specific timestamps to ground your answer.
    If the evidence is insufficient, state that.
    """
    
    try:
        # We reuse the GeminiService wrapper
        answer = gemini_service.generate_answer(prompt)
    except Exception as e:
        answer = "I encountered an error generating the answer."
        print(f"Error generating answer: {e}")
        
    return {"answer": answer, "evidence": evidence}
