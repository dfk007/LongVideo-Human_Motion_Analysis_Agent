# Long-Video Human Motion Analysis Agent

An agentic system that analyzes human physical activity in long-form videos, answering natural language questions about technique, safety, and biomechanics using multimodal AI.

## 🎯 Overview

This system combines computer vision (MediaPipe), semantic search (ChromaDB), and large language models (Ollama Cloud) to provide grounded, evidence-based coaching feedback on human motion.

**Key Features:**
- **Smart Video Processing**: Segments long videos and semantically indexes them for efficient retrieval
- **Pose-Based Analysis**: Extracts 33 3D landmarks, computes angles, velocities, and motion metrics
- **Agentic Reasoning**: LangChain ReAct agent with specialized tools for motion analysis
- **Evidence Grounding**: All answers cite specific timestamps, joint angles, and measurements

## 📺 Live Demo

Experience the agent in action: **[Live Product Walkthrough](https://app.arcade.software/share/K4B0798sL4TgUNtwOl3v)**

## 🏗️ System Architecture

```
┌─────────────┐
│   Video     │ → Segmentation (10s chunks)
│   Upload    │ → Pose Extraction (MediaPipe)
└─────────────┘ → Motion Analysis (angles, velocity)
                → Semantic Indexing (ChromaDB)
       │
       ▼
┌─────────────┐
│   User      │ → "Is this squat safe?"
│   Query     │
└─────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  LangChain ReAct Agent (Ollama)     │
│  ┌──────────────────────────────┐   │
│  │ 1. Search segments           │   │
│  │ 2. Analyze pose data         │   │
│  │ 3. Validate safety           │   │
│  │ 4. Generate grounded answer  │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
       │
       ▼
Answer: "At 12.5s, knee angle is 85° (below 90° safe threshold)"
```

### Design Principles

1. **Perception ≠ Reasoning**: MediaPipe handles pose extraction, LLM handles interpretation
2. **Efficient Retrieval**: Semantic search finds relevant segments (no brute force)
3. **Tool-Based Agent**: Modular tools (search, analyze, validate) vs. monolithic model
4. **Evidence Grounding**: Every claim backed by timestamps and quantitative metrics

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 8GB+ RAM (for MediaPipe processing)

### 1. Clone & Configure

```bash
git clone <your-repo-url>
cd LongVideo-Human_Motion_Analysis_Agent

# Environment is pre-configured with Ollama Cloud
# API key already set in .env
```

### 2. Start Services

```bash
docker compose up -d
```

**Services:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### 3. Upload & Query

**Via UI:**
1. Open http://localhost:3000
2. Upload a video (e.g., squat demonstration)
3. Wait for processing (~1-2 min per minute of video)
4. Ask: *"Is the squat depth sufficient?"*

**Via API:**
```bash
# Upload video
curl -X POST http://localhost:8000/api/videos/upload \
  -F "file=@squat_video.mp4"

# Query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Is this squat safe according to coaching standards?"}'
```

## 📊 Example Queries

| Query | What It Does |
|-------|--------------|
| "Analyze the takeoff technique in this vault" | Finds vault segments, extracts jump angles, analyzes power generation |
| "Is this squat safe?" | Checks knee angle, back angle, depth against safety thresholds |
| "What is the average descent speed?" | Calculates velocity of hip joint during squat descent |
| "Compare rep 1 vs rep 5" | Retrieves both reps, compares joint angles and form degradation |

## 🛠️ Technology Stack

| Layer | Technology | Why? |
|-------|-----------|------|
| **Frontend** | React + Next.js | Modern, responsive UI |
| **Backend** | FastAPI | Async Python, auto-docs |
| **Pose Estimation** | MediaPipe BlazePose | 30+ FPS, easy deployment vs OpenPose |
| **Vector DB** | ChromaDB | Local-first semantic search |
| **LLM** | Ollama Cloud (nemotron-3-nano) | 30B agentic model, cloud-hosted |
| **Agent Framework** | LangChain | ReAct pattern with custom tools |
| **Video Processing** | OpenCV + FFmpeg | Robust video I/O |

## 🎨 Design Decisions

### 1. Why 10-Second Segments?
**Tradeoff**: Most exercises (squat, pushup) complete in 3-8 seconds. 10s captures full reps while maintaining retrieval speed.

**Alternatives Considered:**
- 5s: Too granular, more storage
- 30s: Coarse, slower LLM processing
- Scene detection: Complex, not motion-aligned

### 2. Why MediaPipe Over OpenPose?
**Speed vs Accuracy**: MediaPipe runs 30+ FPS vs OpenPose's 5-10 FPS. Accuracy sacrifice (85% vs 90%) acceptable for coaching use case.

### 3. Why Semantic Retrieval?
**Efficiency**: For a 30-minute video, semantic search queries 5 relevant segments instead of processing 180 segments exhaustively.

### 4. Why Ollama Cloud?
**Deployment Simplicity**: Cloud API eliminates local GPU requirements while providing capable 30B model for complex reasoning.

## 📁 Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── api/           # FastAPI routes
│   │   ├── services/
│   │   │   ├── agent_service.py      # LangChain ReAct agent
│   │   │   ├── pose_estimation.py    # MediaPipe wrapper
│   │   │   ├── motion_analyzer.py    # Biomechanics calculations
│   │   │   └── workflow_service.py   # Video processing pipeline
│   │   └── core/          # Config, database models
│   └── Dockerfile
├── frontend/
│   └── [Next.js app]
├── docker-compose.yml
├── .env                   # Configuration (Ollama API key)
└── DESIGN.md             # Detailed architecture docs
```

## 🧪 Testing

```bash
# Unit tests
cd backend
pytest tests/

# Integration test
pytest tests/integration/test_workflow.py

# Test with example video
curl -X POST http://localhost:8000/api/videos/upload \
  -F "file=@tests/fixtures/squat_sample.mp4"
```

## 📈 Performance

| Metric | Value |
|--------|-------|
| Processing Speed | 1-2s per second of video |
| Query Latency | 3-5 seconds |
| Pose Detection | 90%+ landmark detection |
| Max Video Length | 60 minutes |

**Bottlenecks:**
- Video processing: Pose estimation (can parallelize)
- Query: LLM inference (2-3s with Ollama Cloud)

## 🔒 Security Notes

**Current (Prototype):**
- ⚠️ No authentication
- ⚠️ Videos stored unencrypted
- ⚠️ API key in .env (not production-safe)

**Production Requirements:**
- User authentication (JWT)
- Video encryption at rest
- Secrets management (Vault, K8s secrets)
- RBAC for video access

## 🚧 Known Limitations

1. **Single-person videos**: Multi-person tracking not implemented
2. **No real-time**: Batch processing only (no live streams)
3. **English only**: NLP components not multilingual
4. **Limited sports**: Trained on common movements (squat, vault, running)

## 📚 Documentation

- **[DESIGN.md](./DESIGN.md)**: Detailed architecture, data flow, tradeoffs
- **[API Docs](http://localhost:8000/docs)**: Interactive Swagger UI
- **Demo Video**: [Live Product Walkthrough](https://app.arcade.software/share/K4B0798sL4TgUNtwOl3v)

## 🎬 Demo Video Script

The demo video covers:
1. System overview (architecture diagram)
2. Upload squat video via UI
3. Ask: "Is this squat safe?"
4. Show agent reasoning steps (search → analyze → validate)
5. Result: Grounded answer with timestamps + measurements
6. Explain design choices (perception/reasoning separation, semantic retrieval)

## 🤝 Contributing

This is a take-home project submission. For production use:
- Add multi-person tracking
- Implement real-time analysis pipeline
- Add sport-specific models (gymnastics, Olympic lifting)
- Fine-tune pose model on domain data

## 📄 License

MIT License - see LICENSE file

---

**Contact**: Syed Ali syedeli11@gmail.com  
**Submission Date**: January 2026
