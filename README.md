# Long-Video Human Motion Analysis Agent

A production-grade agentic system for analyzing human physical activity in long-form videos. This system uses a modular architecture that separates perception from reasoning, enabling efficient analysis of sports training, posture correction, gymnastics, and other physical activities.

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Project Overview

This system answers natural language questions about human motion in videos by:
- **Intelligently segmenting** long videos into analyzable chunks.
- **Extracting motion data** using vision-language models for temporal analysis.
- **Reasoning about technique** using LLMs with grounded evidence.
- **Providing explainable answers** with timestamps and visual references.

### Example Queries
- *"Analyze the takeoff technique in this vault."*
- *"Is this squat safe according to coaching standards?"*
- *"What is the subject doing between 0:10 and 0:25?"*

---

## 🏗️ System Architecture

### Design Philosophy

The system follows a **three-phase agentic approach** inspired by human video comprehension:

1. **Retrieve Phase**: Query understanding and relevant segment identification.
2. **Perceive Phase**: Dense motion extraction from selected segments using Gemini 1.5 Flash.
3. **Review Phase**: Reasoning over extracted data to generate grounded answers.

This architecture separates concerns:
- **Perception Layer**: Gemini 1.5 Flash extracts motion and temporal features from video segments.
- **Reasoning Layer**: Gemini 1.5 Flash/Pro interprets the extracted descriptions to answer complex questions.
- **Knowledge Layer**: ChromaDB (Vector Database) stores processed segment descriptions for efficient retrieval.

```mermaid
graph TD
    User[User] -->|Upload Video & Ask Question| Frontend
    Frontend -->|API Request| Backend
    
    subgraph "Backend System"
        Backend -->|Orchestrate| Agent[Gemini Agent]
        
        subgraph "Perception Module"
            VideoProcessor[Video Processor] -->|Chunk & Describe| LLM_Vision[Gemini 1.5 Flash]
            LLM_Vision -->|Embeddings| ChromaDB[(ChromaDB)]
        end
        
        subgraph "Reasoning Module"
            Agent -->|Search| ChromaDB
            Agent -->|Synthesize Answer| LLM_Reasoning[Gemini 1.5 Flash]
        end
    end
    
    Agent -->|Response + Timestamps| Frontend
```

---

## 🔄 Data Flow & Storage

### 1. Storage & Pre-processing
- **Uploads**: Raw videos are stored in `backend/uploads/`.
- **Processing**: Videos are automatically split into 15-second chunks using FFmpeg and stored in `backend/processed/<video_name>/`.

### 2. Ingestion (Perception)
- Each 15s chunk is analyzed by **Gemini 1.5 Flash** to generate a dense text description of movement, safety, and mechanics.
- These descriptions are converted into vectors and stored in **ChromaDB**.

### 3. Retrieval (Reasoning)
- When a user asks a question, the system performs a **semantic search** in ChromaDB to find the most relevant video segments.
- The retrieved descriptions are used as "evidence" for the LLM to generate a grounded, timestamped answer.

---

## 🔧 Technology Stack

### Core Technologies
- **Backend**: FastAPI (Python 3.11+)
- **Frontend**: Next.js + TypeScript + Tailwind CSS
- **Vector Database**: ChromaDB
- **Database**: PostgreSQL (for session/metadata)
- **Containerization**: Docker + Docker Compose

### AI/ML Models
- **Video Understanding**: Google Gemini 1.5 Flash (via Vertex AI)
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- **Google Cloud Project** with Vertex AI API enabled.
- Service Account credentials in `backend/credentials.json`.

### Installation & Running

1. **Configure Environment**:
   Update `.env` with your project details:
   ```env
   GOOGLE_CLOUD_PROJECT=your-project-id
   GOOGLE_CLOUD_LOCATION=us-central1
   VERTEX_AI_MODEL=gemini-1.5-flash
   ```

2. **Launch System**:
   ```bash
   docker compose up --build
   ```

3. **Access**:
   - Frontend: `http://localhost:3000`
   - Backend API: `http://localhost:8000/docs`

---

## 🔑 Key Design Decisions

### 1. **Separation of Perception and Reasoning**
**Why**: Vision models are good at *what they see*, LLMs are good at *understanding what it means*
- **Perception**: MediaPipe/Gemini extracts raw pose data and descriptions (no interpretation)
- **Reasoning**: Gemini interprets motion data in context of coaching standards
- **Benefit**: Modular, testable, and easier to improve each component independently

### 2. **Hierarchical Video Segmentation**
**Why**: Processing 30-minute videos frame-by-frame is computationally prohibitive
- **Approach**: 15-second segments stored with embeddings
- **Retrieval**: Only analyze segments relevant to the query
- **Tradeoff**: May miss cross-segment patterns (addressed via context expansion)

### 3. **Vector Database for Temporal Search**
**Why**: Traditional search (timestamps, keywords) doesn't capture semantic meaning
- **Solution**: ChromaDB with sentence transformer embeddings
- **Benefit**: Query "squat depth" retrieves relevant segments even without exact keywords
- **Example**: "bad form" matches segments with poor technique descriptions

### 4. **Grounding in Evidence**
**Why**: Generic answers aren't useful for coaching
- **Requirement**: Every claim must reference timestamps + metrics
- **Implementation**: Prompt engineering forces model to cite evidence
- **Example**: "At 2:15, knee angle is 110° (should be 90°±5°)"

---

## 📊 Performance Considerations

### Computational Efficiency
- **Video Processing**: ~1-2 seconds per second of video
- **Query Response**: 2-5 seconds (retrieval + reasoning)
- **Memory**: ~500MB per 10-minute video in memory

### Optimization Strategies
1. **Lazy Loading**: Only process segments when needed
2. **Caching**: Store pose data to avoid reprocessing
3. **Model Selection**: Use `gemini-1.5-flash` for speed, `gemini-1.5-pro` for accuracy

---

## 🧪 Testing & Validation

### Test Cases
```bash
# Test video processing
python backend/tests/test_video_processing.py

# Test LLM integration
python backend/tests/test_gemini.py
```

### Validation Metrics
- **Retrieval Quality**: Relevance of retrieved segments to query
- **Answer Grounding**: % of answers with timestamp references

---

## 🛣️ Future Improvements

### Short-term
- [ ] Add confidence scores to all outputs
- [ ] Implement video quality checks
- [ ] Support multi-person videos

### Medium-term
- [ ] Fine-tune smaller models for specific sports
- [ ] Add comparative analysis (rep-to-rep)
- [ ] Implement trajectory tracking

### Long-term
- [ ] Real-time analysis for live coaching
- [ ] Mobile app deployment
- [ ] Integration with wearable sensors

---

## 📝 Project Structure

```
motion-analysis-agent/
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI endpoints
│   │   ├── services/         # Core business logic
│   │   │   ├── video_service.py      # Segmentation & FFmpeg
│   │   │   ├── workflow.py           # Orchestration
│   │   │   ├── vector_db.py          # ChromaDB wrapper
│   │   │   └── gemini_service.py     # Vertex AI integration
│   │   ├── core/             # Configuration
│   │   └── main.py           # Entry point
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app/                  # Next.js App Router
│   ├── components/           # React components
│   └── package.json
├── docker-compose.yml
└── README.md                 # This file
```

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Google Gemini**: Multi-modal video understanding
- **ChromaDB**: Vector storage and retrieval
- **FastAPI**: High-performance backend framework
- **Next.js**: Modern React framework
