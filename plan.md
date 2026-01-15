# Plan: Long-Video Human Motion Analysis Agent

## 1. Project Goal
Build an agentic system capable of analyzing long-form videos of human physical activity (sports, exercises) to answer natural language questions about technique, safety, and correctness using "Gemini 2.5/3" (latest Vertex AI models).

## 2. deliverables & Expectations (from PDF)
- **GitHub Repository**: Runnable code (Dockerized) & Documentation.
- **Key Capabilities**:
    - Handle long videos (efficiently, not brute force).
    - Separate perception (seeing) from reasoning (thinking).
    - Ground answers in evidence (timestamps, metrics).
    - Use agentic tools/planning.
- **Demo Video**: (To be recorded by user later).

## 3. Architecture Overview
- **Frontend**: Next.js (React) - Minimal chat interface + Video Player.
- **Backend**: FastAPI (Python) - Main agent logic, API endpoints.
- **Database**: PostgreSQL (Metadata, Chat History).
- **Vector Database**: ChromaDB (Video embeddings, frame descriptions for retrieval).
- **AI/LLM Provider**: Google Vertex AI (Gemini Models).

## 4. Implementation Steps

### Phase 1: Infrastructure (Current Step)
- [x] Analyze Requirements.
- [x] Create `docker-compose.yml`.
- [x] Create `.env`.
- [ ] Initialize Frontend (Next.js) structure.
- [ ] Initialize Backend (FastAPI) structure.

### Phase 2: Backend - Perception & Ingestion
- **Video Ingestion Pipeline**:
    - Upload video endpoint.
    - **Preprocessing**: Split long video into manageable chunks (e.g., 10-30s).
    - **Indexing**:
        - Generate text descriptions for chunks (using Gemini Flash for speed).
        - Embed descriptions/frames into **ChromaDB**.
    - **Pose Extraction (Tool)**:
        - Integrate MediaPipe or similar to extract skeletal keypoints for specific segments when required (for "safety" analysis).

### Phase 3: Backend - Agentic Logic
- **Agent Framework**: Use LangChain or raw Python tool calling with Gemini.
- **Tools Definition**:
    - `search_video_segments(query)`: Retrieve relevant timestamps from ChromaDB.
    - `analyze_pose_safety(timestamp_start, timestamp_end)`: Detailed analysis of specific frames.
    - `get_coaching_guidelines(activity_type)`: Retrieve standard safety rules.
- **Reasoning Loop**:
    1. Receive user query (e.g., "Is the squat depth sufficient?").
    2. Plan: Identify need to see "squat" segments.
    3. Tool: `search_video_segments("squat bottom position")`.
    4. Observation: Get timestamps [00:15, 02:30].
    5. Tool: `analyze_pose_safety(00:15, 00:20)`.
    6. Final Answer: Combine observations into a user-friendly response.

### Phase 4: Frontend Implementation
- Chat interface to send text query.
- Video player that syncs with Agent's cited timestamps (e.g., clicking a timestamp in the answer jumps the video).
- File upload for the video.

### Phase 5: Testing & Refinement
- Test with sample workout videos.
- Verify "long video" efficiency (ensure we don't feed the whole hour to the LLM context window blindly).
