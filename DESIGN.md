# System Design Document: Long-Video Human Motion Analysis Agent

**Version**: 1.0  
**Last Updated**: January 2026  
**Author**: [Your Name]

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Requirements](#system-requirements)
3. [Architecture Overview](#architecture-overview)
4. [Detailed Component Design](#detailed-component-design)
5. [Data Flow](#data-flow)
6. [Design Tradeoffs](#design-tradeoffs)
7. [Performance Analysis](#performance-analysis)
8. [Future Architecture Evolution](#future-architecture-evolution)

---

## 1. Executive Summary

This document describes the architecture of a production-grade agentic system for analyzing human motion in long-form videos. The system is designed for sports coaching, physical therapy, and movement analysis applications.

### Key Design Principles

1. **Modularity**: Clear separation between perception, reasoning, and retrieval layers
2. **Efficiency**: Smart segment retrieval instead of exhaustive analysis
3. **Explainability**: All outputs grounded in timestamps and quantitative metrics
4. **Extensibility**: Easy to add new sports, pose models, or reasoning strategies

### System Capabilities

- Analyze videos of 1-60 minutes with human physical activity
- Answer natural language questions about technique, safety, and form
- Provide timestamp-referenced, metric-grounded explanations
- Process multiple types of movements: squats, gymnastics, running, etc.

---

## 2. System Requirements

### Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR1 | Accept video upload in common formats (MP4, AVI, MOV) | Must Have |
| FR2 | Process natural language queries about video content | Must Have |
| FR3 | Extract human pose landmarks from video frames | Must Have |
| FR4 | Calculate joint angles, velocities, and motion metrics | Must Have |
| FR5 | Retrieve relevant video segments based on query semantics | Must Have |
| FR6 | Generate grounded answers with timestamps and metrics | Must Have |
| FR7 | Visualize pose data and motion analysis results | Should Have |
| FR8 | Support multi-person videos | Could Have |
| FR9 | Provide confidence scores for all outputs | Should Have |
| FR10 | Enable comparative analysis (e.g., rep 1 vs rep 5) | Should Have |

### Non-Functional Requirements

| ID | Requirement | Target | Priority |
|----|-------------|--------|----------|
| NFR1 | Video processing latency | < 2s per second of video | High |
| NFR2 | Query response time | < 5 seconds | High |
| NFR3 | Pose estimation accuracy | > 90% landmark detection | High |
| NFR4 | System availability | > 99% uptime | Medium |
| NFR5 | Concurrent users supported | 10+ simultaneous users | Medium |
| NFR6 | Maximum video length | 60 minutes | Medium |
| NFR7 | API response format | JSON with clear structure | High |
| NFR8 | Error handling | Graceful degradation | High |

### Technical Constraints

- **Computational**: Single-server deployment (no distributed processing yet)
- **Storage**: Local filesystem for videos (no cloud storage integration)
- **Authentication**: No user authentication system (future enhancement)
- **Real-time**: No live streaming support (batch processing only)

---

## 3. Architecture Overview

### 3.1 High-Level Architecture

The system follows a **layered architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                   Presentation Layer                         │
│  React Frontend + REST API                                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                   Application Layer                          │
│  Query Router → Agent Orchestrator → Response Builder        │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                   Business Logic Layer                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │  Retrieval │  │ Perception │  │  Reasoning │            │
│  │   Agent    │→ │   Engine   │→ │   Agent    │            │
│  └────────────┘  └────────────┘  └────────────┘            │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                    Data Layer                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │  Video   │  │  Vector  │  │  Pose    │                  │
│  │  Store   │  │    DB    │  │  Cache   │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Core Components

#### 3.2.1 Perception Layer
- **Video Processor**: Segments videos into analyzable chunks
- **Pose Estimator**: Extracts skeletal landmarks using MediaPipe
- **Motion Analyzer**: Computes angles, velocities, accelerations

#### 3.2.2 Knowledge Layer
- **Vector Database**: ChromaDB for semantic segment storage
- **Embedding Generator**: Sentence transformers for text embeddings
- **Metadata Store**: Pose data, timestamps, quality metrics

#### 3.2.3 Reasoning Layer
- **Query Understanding**: Parse user intent and extract key concepts
- **Segment Retrieval**: Find relevant video portions via semantic search
- **Answer Generation**: LLM synthesizes motion data into explanations

---

## 4. Detailed Component Design

### 4.1 Video Processing Pipeline

#### 4.1.1 Segmentation Strategy

**Goal**: Break long videos into manageable, semantically meaningful chunks

**Implementation**:
```python
def segment_video(video_path: str, segment_duration: int = 10) -> List[Segment]:
    """
    Segments video into fixed-duration chunks with overlap
    
    Args:
        video_path: Path to input video
        segment_duration: Duration of each segment in seconds
    
    Returns:
        List of Segment objects with start/end times
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    segments = []
    overlap = 2  # 2-second overlap between segments
    
    for start in range(0, int(duration), segment_duration - overlap):
        end = min(start + segment_duration, duration)
        segments.append(Segment(
            start_time=start,
            end_time=end,
            start_frame=int(start * fps),
            end_frame=int(end * fps)
        ))
    
    return segments
```

**Design Rationale**:
- **Fixed duration**: 10 seconds balances granularity vs. processing cost
- **Overlap**: 2-second overlap prevents missing actions at boundaries
- **Alternative considered**: Scene-based segmentation (rejected due to complexity)

#### 4.1.2 Frame Sampling

**Challenge**: Processing every frame is computationally expensive

**Solution**: Intelligent frame sampling based on motion
```python
def sample_frames(segment: Segment, sampling_rate: int = 3) -> List[Frame]:
    """
    Sample frames from segment at specified rate
    
    For 30fps video with sampling_rate=3, processes 10fps
    For high-motion segments, may increase sampling
    """
    frames = []
    for frame_idx in range(segment.start_frame, segment.end_frame, sampling_rate):
        frame = extract_frame(video_path, frame_idx)
        frames.append(frame)
    return frames
```

**Tradeoff**: 10fps sampling vs. full 30fps
- **Benefit**: 3x faster processing, sufficient for most human motion (< 5 Hz)
- **Cost**: May miss very fast movements (e.g., tennis serve)
- **Mitigation**: Increase sampling rate for high-speed sports

### 4.2 Pose Estimation

#### 4.2.1 MediaPipe Integration

**Choice**: MediaPipe Pose (BlazePose model)

**Why MediaPipe?**
| Feature | MediaPipe | OpenPose | AlphaPose |
|---------|-----------|----------|-----------|
| Speed | 30+ FPS | 5-10 FPS | 10-15 FPS |
| Accuracy | 85-90% | 90-95% | 88-92% |
| Ease of Use | High | Medium | Medium |
| Deployment | Easy | Complex | Medium |
| License | Apache 2.0 | Academic | Academic |

**Decision**: MediaPipe chosen for speed and ease of deployment

#### 4.2.2 Landmark Extraction

```python
class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=lite, 1=full, 2=heavy
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_pose(self, frame: np.ndarray) -> PoseData:
        """Extract 33 3D landmarks from frame"""
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks:
            return None
        
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append({
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility
            })
        
        return PoseData(
            landmarks=landmarks,
            timestamp=frame.timestamp,
            confidence=np.mean([lm.visibility for lm in results.pose_landmarks.landmark])
        )
```

**Key Parameters**:
- `model_complexity=1`: Balances accuracy and speed
- `smooth_landmarks=True`: Reduces jitter in tracking
- `min_detection_confidence=0.5`: Threshold for valid pose detection

### 4.3 Motion Analysis

#### 4.3.1 Angle Calculation

```python
def calculate_angle(p1: Point3D, p2: Point3D, p3: Point3D) -> float:
    """
    Calculate angle at p2 formed by p1-p2-p3
    
    Used for knee angle, elbow angle, hip angle, etc.
    """
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg
```

**Common Angles Computed**:
- **Knee Angle**: Hip → Knee → Ankle
- **Hip Angle**: Shoulder → Hip → Knee
- **Elbow Angle**: Shoulder → Elbow → Wrist
- **Back Angle**: Hip → Shoulder → vertical

#### 4.3.2 Velocity & Acceleration

```python
def calculate_velocity(pose_sequence: List[PoseData], 
                      landmark_idx: int) -> List[float]:
    """
    Calculate velocity of specific landmark over time
    
    velocity[t] = (position[t+1] - position[t]) / dt
    """
    velocities = []
    for i in range(len(pose_sequence) - 1):
        p1 = pose_sequence[i].landmarks[landmark_idx]
        p2 = pose_sequence[i+1].landmarks[landmark_idx]
        dt = pose_sequence[i+1].timestamp - pose_sequence[i].timestamp
        
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        dz = p2.z - p1.z
        
        velocity = np.sqrt(dx**2 + dy**2 + dz**2) / dt
        velocities.append(velocity)
    
    return velocities
```

### 4.4 Semantic Retrieval

#### 4.4.1 Vector Database Schema

**ChromaDB Collection Structure**:
```python
collection_schema = {
    "name": "video_segments",
    "metadata": {
        "video_id": str,
        "segment_id": str,
        "start_time": float,
        "end_time": float,
        "duration": float,
        "num_frames": int,
        "avg_confidence": float
    },
    "documents": [
        "Text description of segment motion"
    ],
    "embeddings": [
        384-dimensional vector from sentence-transformers
    ]
}
```

**Document Format**:
```
"Segment from 10.0s to 20.0s: Person performing squat motion. 
Knee angle ranges from 180° (standing) to 85° (bottom position). 
Hip descends to 0.45m height. Back maintains 15° forward lean. 
Movement velocity is moderate (0.3 m/s descent)."
```

#### 4.4.2 Retrieval Algorithm

```python
def retrieve_relevant_segments(query: str, 
                               k: int = 5,
                               confidence_threshold: float = 0.7) -> List[Segment]:
    """
    Retrieve top-k segments most relevant to query
    
    Args:
        query: Natural language question
        k: Number of segments to retrieve
        confidence_threshold: Minimum similarity score
    
    Returns:
        List of relevant segments with similarity scores
    """
    # Generate query embedding
    query_embedding = embedding_model.encode(query)
    
    # Search vector database
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k * 2,  # Over-retrieve and filter
        include=['documents', 'metadatas', 'distances']
    )
    
    # Filter by confidence and return top-k
    filtered_results = [
        r for r in results 
        if r['similarity'] >= confidence_threshold
    ]
    
    return filtered_results[:k]
```

### 4.5 Reasoning Agent

#### 4.5.1 Prompt Engineering Strategy

**System Prompt**:
```
You are an expert sports coach and biomechanics analyst. Analyze human 
motion data and provide coaching feedback.

CRITICAL RULES:
1. ALWAYS cite timestamps for every claim
2. ALWAYS include numerical metrics (angles, velocities, heights)
3. Reference coaching standards when discussing safety/technique
4. If data is insufficient, say so explicitly
5. Use clear, actionable language

OUTPUT FORMAT:
{
  "answer": "Main answer to the question",
  "evidence": [
    {
      "timestamp": 12.5,
      "observation": "Knee angle is 85°",
      "assessment": "Below safe threshold of 90°"
    }
  ],
  "recommendation": "Actionable coaching advice",
  "confidence": 0.85
}
```

#### 4.5.2 Multi-Step Reasoning

**For complex queries, the agent follows a chain-of-thought process**:

```python
def answer_complex_query(query: str, segments: List[Segment]) -> Answer:
    """
    Multi-step reasoning for complex motion questions
    
    Example: "Is this squat safe according to coaching standards?"
    
    Step 1: Identify safety criteria
    Step 2: Extract relevant metrics from segments
    Step 3: Compare against standards
    Step 4: Synthesize judgment
    """
    
    # Step 1: Decompose query into sub-questions
    sub_questions = decompose_query(query)
    # ["What is the knee angle?", "What is the back angle?", "Is depth sufficient?"]
    
    # Step 2: Answer each sub-question
    sub_answers = []
    for sq in sub_questions:
        relevant_data = extract_motion_data(segments, sq)
        answer = llm.generate(sq, relevant_data)
        sub_answers.append(answer)
    
    # Step 3: Synthesize final answer
    final_answer = llm.synthesize(query, sub_answers)
    
    return final_answer
```

---

## 5. Data Flow

### 5.1 Video Upload Flow

```
User uploads video (MP4)
    ↓
Backend receives file → Save to /uploads
    ↓
Video Processor triggered
    ↓
1. Extract metadata (duration, FPS, resolution)
2. Segment into 10s chunks with 2s overlap
    ↓
For each segment:
    a. Sample frames at 10 FPS
    b. Extract pose landmarks (MediaPipe)
    c. Calculate angles, velocities
    d. Generate text description
    e. Create embedding
    f. Store in ChromaDB
    ↓
Return processing complete → Frontend shows ready state
```

**Time Complexity**: O(n) where n = video duration
**Space Complexity**: O(m) where m = number of segments

### 5.2 Query Processing Flow

```
User submits question: "Is the squat form correct?"
    ↓
Query Understanding Agent
    - Extract key concepts: ["squat", "form", "correctness"]
    - Identify required metrics: [knee_angle, back_angle, depth]
    ↓
Retrieval Agent queries ChromaDB
    - Search for segments matching "squat form"
    - Return top 5 most relevant segments
    ↓
Perception Engine
    - Load pose data for retrieved segments
    - Compute required angles, positions
    - Organize into structured JSON
    ↓
Reasoning Agent (LLM)
    - Analyze pose data against coaching standards
    - Generate explanation with timestamps
    - Provide safety assessment and recommendations
    ↓
Response Builder
    - Format answer with evidence
    - Add visualizations (angle graphs, pose overlays)
    - Return to frontend
```

**Average Latency Breakdown**:
- Retrieval: 0.5s
- Pose data loading: 0.2s
- LLM inference: 2-3s
- **Total**: ~3s

---

## 6. Design Tradeoffs

### 6.1 Segment Duration: 10 seconds

**Options Considered**:
| Duration | Pros | Cons | Decision |
|----------|------|------|----------|
| 5 seconds | Finer granularity, faster retrieval | More segments, higher storage | ❌ |
| **10 seconds** | **Good balance, captures full reps** | **May span multiple actions** | ✅ |
| 30 seconds | Fewer segments, less storage | Coarse granularity, slower LLM processing | ❌ |

**Rationale**: Most human movements (squat, pushup, vault) complete in 3-8 seconds. 10-second segments capture full actions while maintaining reasonable segment count.

### 6.2 Pose Model: MediaPipe vs. OpenPose

| Factor | MediaPipe | OpenPose | Decision |
|--------|-----------|----------|----------|
| **Speed** | 30-60 FPS | 5-10 FPS | ✅ MediaPipe |
| **Accuracy** | 85-90% | 90-95% | |
| **Deployment** | pip install | Complex build | ✅ MediaPipe |
| **Hand/Face** | Separate models | Integrated | |

**Tradeoff**: Sacrificed 5% accuracy for 3-6x speed and easier deployment.

**Mitigation**: Use confidence scores to flag low-quality poses.

### 6.3 Embedding Model: Sentence-BERT vs. OpenAI Embeddings

| Model | Dimensions | Speed | Cost | Decision |
|-------|------------|-------|------|----------|
| **all-MiniLM-L6-v2** | 384 | Fast (local) | Free | ✅ |
| OpenAI ada-002 | 1536 | Slow (API) | $0.0001/1K tokens | ❌ |

**Rationale**: Local embedding model eliminates API costs and latency for production deployment.

### 6.4 Reasoning: Local LLM vs. Cloud API

**Current**: Google Gemini 1.5 Flash (cloud API)

**Future Consideration**: Qwen2-VL (open-source, local)

| Factor | Gemini API | Qwen2-VL Local | Current Choice |
|--------|-----------|----------------|----------------|
| **Speed** | 2-3s | 5-10s | ✅ Gemini |
| **Cost** | $0.35/1M tokens | Hardware only | |
| **Quality** | Excellent | Very good | ✅ Gemini |
| **Privacy** | Data sent to Google | Fully local | |

**Rationale for Prototype**: Cloud API for rapid development, with path to local model for production.

---

## 7. Performance Analysis

### 7.1 Scalability Characteristics

#### Current Bottlenecks
1. **Video Processing**: 1-2 seconds per second of video (pose estimation)
2. **Storage**: Embeddings grow linearly with video duration
3. **Query Latency**: Dominated by LLM inference time

#### Projected Performance

| Video Duration | Processing Time | Storage | Query Time |
|----------------|-----------------|---------|------------|
| 1 minute | 1-2 minutes | 50 MB | 3s |
| 10 minutes | 10-20 minutes | 500 MB | 3-4s |
| 30 minutes | 30-60 minutes | 1.5 GB | 3-5s |
| 60 minutes | 60-120 minutes | 3 GB | 3-5s |

**Note**: Query time is relatively constant due to selective retrieval

### 7.2 Optimization Opportunities

#### Short-term (Current Sprint)
1. **Parallel Frame Processing**: Process frames in batches using multiprocessing
2. **Pose Data Caching**: Cache extracted poses to avoid reprocessing
3. **Smart Segment Selection**: Use motion analysis to skip static segments

#### Medium-term (Next Release)
1. **GPU Acceleration**: Move pose estimation to GPU for 5-10x speedup
2. **Incremental Processing**: Process new videos incrementally, not all at once
3. **Compression**: Compress pose data using delta encoding

#### Long-term (Future Versions)
1. **Distributed Processing**: Use Celery + Redis for async video processing
2. **Cloud Storage**: Offload video files to S3/GCS
3. **Edge Deployment**: Run pose estimation on edge devices (mobile, webcam)

---

## 8. Future Architecture Evolution

### 8.1 Phase 2: Real-Time Analysis

**Goal**: Analyze live video streams for instant coaching feedback

**Architecture Changes**:
```
Current: Batch Processing
Video → Segments → Pose Extraction → Storage → Query

Future: Streaming Pipeline
Video Stream → Real-time Pose Extraction → Incremental Analysis → Live Feedback
               ↓
         Ring Buffer (last 30s)
               ↓
         Trigger Alerts on Form Issues
```

**Key Technologies**:
- **WebRTC**: Low-latency video streaming
- **Redis Streams**: Temporal buffer for recent frames
- **Rule Engine**: Fast heuristics for immediate feedback

### 8.2 Phase 3: Multi-Modal Analysis

**Enhancement**: Combine pose + audio + equipment tracking

**New Inputs**:
- **Audio**: Coach verbal cues, athlete breathing patterns
- **Equipment**: Barbell tracking, resistance band tension
- **Wearables**: Heart rate, muscle activation (EMG)

**Architecture**:
```
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  Video   │  │  Audio   │  │Equipment │  │ Wearables│
│  Stream  │  │  Stream  │  │ Tracking │  │   Data   │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │              │              │
     └─────────────┴──────────────┴──────────────┘
                        │
                ┌───────┴────────┐
                │ Multi-Modal    │
                │ Fusion Engine  │
                └───────┬────────┘
                        │
                ┌───────┴────────┐
                │  Comprehensive │
                │    Analysis    │
                └────────────────┘
```

### 8.3 Phase 4: Personalized Coaching

**Goal**: Adapt analysis to individual athlete profiles

**New Components**:
- **Athlete Profiles**: Physical attributes, injury history, goals
- **Progress Tracking**: Longitudinal motion analysis over weeks/months
- **Adaptive Models**: Fine-tuned models per athlete or sport
- **Feedback Loop**: Incorporate coach annotations to improve system

---

## 9. Security & Privacy Considerations

### 9.1 Current Implementation
- **No user authentication**: All uploaded videos are public
- **No encryption**: Videos stored in plaintext
- **No audit logs**: No tracking of who accessed what

### 9.2 Production Requirements

#### Must-Have for Production
1. **User Authentication**: JWT-based auth with role-based access control
2. **Video Encryption**: Encrypt at rest using AES-256
3. **Access Control**: Users can only access their own videos
4. **Audit Logging**: Track all video uploads, queries, and accesses
5. **GDPR Compliance**: Allow users to delete their data
6. **Rate Limiting**: Prevent abuse of API endpoints

#### Data Privacy Architecture
```
User Video Upload
    ↓
Encrypt (AES-256) → Store encrypted
    ↓
Associate with User ID (FK)
    ↓
Access Control: Check user.id == video.user_id
    ↓
Decrypt only when authorized user requests
```

---

## 10. Testing Strategy

### 10.1 Unit Tests
- **Video Processor**: Test segmentation, frame extraction
- **Pose Estimator**: Test landmark extraction, confidence filtering
- **Motion Analyzer**: Test angle calculations, velocity computation
- **Retrieval**: Test embedding generation, similarity search

### 10.2 Integration Tests
- **End-to-End**: Upload video → Process → Query → Verify answer
- **API Tests**: Test all REST endpoints with various inputs
- **Error Handling**: Test invalid videos, corrupted files, missing data

### 10.3 Performance Tests
- **Load Testing**: 10 concurrent video uploads + 50 concurrent queries
- **Stress Testing**: Maximum video duration (60 minutes)
- **Latency**: Measure P50, P95, P99 query response times

### 10.4 Validation Tests
- **Pose Accuracy**: Compare MediaPipe outputs against ground truth datasets
- **Answer Quality**: Human evaluation of answer relevance and grounding
- **Retrieval Quality**: Measure precision@K and recall@K for segment retrieval

---

## 11. Deployment Strategy

### 11.1 Development
- **Docker Compose**: Backend + Frontend + ChromaDB
- **Local API Keys**: Gemini API key in `.env`
- **Hot Reload**: FastAPI auto-reload, React dev server

### 11.2 Staging
- **Kubernetes**: Deploy to local K8s cluster (minikube)
- **Secret Management**: Use K8s secrets for API keys
- **Monitoring**: Prometheus + Grafana for metrics

### 11.3 Production (Future)
- **Cloud Platform**: GCP or AWS
- **Container Orchestration**: GKE or EKS
- **Object Storage**: GCS or S3 for videos
- **Database**: Cloud-hosted ChromaDB or Pinecone
- **CDN**: CloudFlare for frontend assets
- **Monitoring**: Datadog or New Relic
- **CI/CD**: GitHub Actions → Build → Test → Deploy

---

## 12. Conclusion

This architecture balances:
- **Speed**: Fast enough for interactive coaching (3-5s queries)
- **Accuracy**: Good enough for practical coaching feedback (85-90%)
- **Simplicity**: Easy to understand, test, and extend
- **Scalability**: Clear path from prototype to production

The key insight is **separating perception from reasoning**: let computer vision models do what they're good at (extracting pose), and let LLMs do what they're good at (understanding meaning and coaching standards).

---

## Appendix A: Key Libraries & Versions

```
# Core Dependencies
fastapi==0.104.1
opencv-python==4.8.1.78
mediapipe==0.10.8
chromadb==0.4.18
sentence-transformers==2.2.2
google-generativeai==0.3.1
langchain==0.1.0
numpy==1.24.3
scipy==1.11.4
```

## Appendix B: API Specification

See `API.md` for detailed REST API documentation.

## Appendix C: Coaching Standards Database

See `COACHING_STANDARDS.md` for biomechanical safety thresholds.

---

**Document Status**: Draft v1.0  
**Review Status**: Pending Peer Review  
**Next Update**: After Phase 2 Implementation
