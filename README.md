# Long-Video Human Motion Analysis Agent
## 🎯 Project Overview | [Demo Video](https://app.arcade.software/share/So5c4zYey9EO218oKDI2)

A production-grade agentic system for analyzing human motion in long videos. This system fulfills the requirements of separating perception from reasoning, handling long videos efficiently, and providing evidence-grounded answers.

## Architecture

The system follows a modular architecture:

1.  **Perception Layer**: 
    *   **MediaPipe Pose**: Extracts 33 3D skeletal landmarks from video frames.
    *   **Motion Analyzer**: Calculates biomechanically meaningful angles (knee flexion, hip hinge, etc.) and detects safety violations based on established guidelines.

2.  **Reasoning Layer**:
    *   **LangChain Agent**: A ReAct agent that plans analysis steps.
    *   **Tools**:
        *   `search_video`: Finds relevant video segments using semantic search (ChromaDB).
        *   `analyze_motion`: Retrieves quantitative pose data for specific segments.
        *   `get_safety_guidelines`: Accesses the knowledge base of biomechanics standards.

3.  **Video Pipeline**:
    *   FFmpeg splits long videos into manageable 15s chunks.
    *   Gemini 1.5 Flash generates semantic descriptions for each chunk.
    *   Pose data is pre-computed and stored for fast retrieval.

## Features

*   **Evidence Grounding**: Answers cite specific timestamps ("at 2:31"), quantitative measurements ("knee angle: 85°"), and authoritative sources ("NSCA Guidelines").
*   **Efficient Processing**: Analyzes long videos (10+ mins) by processing chunks in parallel (conceptually) and using vector search to focus on relevant parts.
*   **Safety Analysis**: Validates movements against a structured database of safety standards for Squats, Deadlifts, Pushups, etc.

## Setup

1.  **Prerequisites**: Docker and Docker Compose.
2.  **Configuration**:
    ```bash
    cp .env.example .env
    # Edit .env and add your GOOGLE_API_KEY
    ```
3.  **Run**:
    ```bash
    ./verify_setup.sh  # Optional: Check your setup
    docker compose up -d --build
    ```
4.  **Access**:
    *   Frontend: http://localhost:3000
    *   Backend Docs: http://localhost:8000/docs

## Usage

1.  Upload a video (e.g., a 5-minute squat tutorial).
2.  Wait for processing (extraction of poses and indexing).
3.  Ask questions:
    *   "Is the squat depth sufficient?"
    *   "Analyze the form errors in the third set."
    *   "Are the knees tracking correctly?"

## Design Choices

*   **MediaPipe vs. LLM Vision**: We use MediaPipe for quantitative metrics because LLMs are not yet precise enough for angle measurements. We use Gemini for high-level semantic understanding (context, intent).
*   **Chunking**: 15s chunks balance granularity (capturing single reps) with retrieval efficiency.
*   **Agent Pattern**: The ReAct agent allows the system to "look" before it "leaps" - searching for relevant content before performing expensive detailed analysis.
