# Long-Video Human Motion Analysis - Project Setup Instructions

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [System Requirements](#system-requirements)
3. [Installation Steps](#installation-steps)
4. [Project Structure](#project-structure)
5. [Configuration](#configuration)
6. [Testing the Setup](#testing-the-setup)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Accounts
1. **Anthropic Account** - For Claude API
   - Sign up at: https://console.anthropic.com
   - Get API key from dashboard
   
2. **Google AI Studio Account** - For Gemini API
   - Sign up at: https://aistudio.google.com
   - Get API key from API Keys section

### Required Software
- **Python**: Version 3.9 or higher
- **FFmpeg**: For video processing
- **Git**: For version control
- **pip**: Python package manager

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Ubuntu 20.04+
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space
- **CPU**: Multi-core processor (4+ cores recommended)
- **GPU**: Optional (speeds up MediaPipe)

### Recommended for Production
- **RAM**: 16GB+
- **Storage**: 50GB+ SSD
- **CPU**: 8+ cores
- **GPU**: NVIDIA GPU with CUDA support

---

## Installation Steps

### Step 1: Install System Dependencies

#### On Ubuntu/Debian
```bash
# Update package list
sudo apt update

# Install FFmpeg
sudo apt install ffmpeg

# Install Python development tools
sudo apt install python3-dev python3-pip

# Install system libraries for MediaPipe
sudo apt install libgl1-mesa-glx libglib2.0-0
```

#### On macOS
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install FFmpeg
brew install ffmpeg

# Install Python (if needed)
brew install python@3.11
```

#### On Windows
1. **Install FFmpeg**:
   - Download from: https://ffmpeg.org/download.html
   - Extract to `C:\ffmpeg`
   - Add `C:\ffmpeg\bin` to PATH environment variable

2. **Install Python**:
   - Download from: https://www.python.org/downloads/
   - Run installer, check "Add Python to PATH"

---

### Step 2: Clone or Create Project Directory
```bash
# Create project directory
mkdir motion-analysis-agent
cd motion-analysis-agent

# Initialize git repository
git init
```

---

### Step 3: Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

---

### Step 4: Install Python Dependencies

#### Create requirements.txt
```bash
# Create requirements.txt file
touch requirements.txt
```

#### Add the following to requirements.txt:
```
# Core dependencies
python-dotenv==1.0.0
pydantic==2.5.0

# Video processing
opencv-python==4.8.1.78
moviepy==1.0.3

# Pose estimation
mediapipe==0.10.8

# CLIP and embeddings
transformers==4.36.0
torch==2.1.0
pillow==10.1.0

# Vector database
chromadb==0.4.18

# LangChain ecosystem
langchain==0.1.0
langchain-anthropic==0.1.0
langchain-google-genai==0.0.5
langchain-community==0.0.10

# API clients
anthropic==0.8.0
google-generativeai==0.3.1

# Utilities
numpy==1.24.3
pandas==2.1.4
tqdm==4.66.1
requests==2.31.0
```

#### Install all packages
```bash
pip install -r requirements.txt
```

---

### Step 5: Verify FFmpeg Installation
```bash
# Check FFmpeg version
ffmpeg -version

# Expected output: FFmpeg version 4.x or higher
```

---

### Step 6: Setup API Keys

#### Create .env file
```bash
# Create .env file in project root
touch .env
```

#### Add your API keys to .env:
```
# Anthropic Claude API
ANTHROPIC_API_KEY=your_claude_api_key_here

# Google Gemini API
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional: Set model names
CLAUDE_MODEL=claude-sonnet-4-20250514
GEMINI_MODEL=gemini-2.0-flash
```

#### Create .env.example (for version control)
```
ANTHROPIC_API_KEY=sk-ant-xxxxx
GOOGLE_API_KEY=AIzaSyxxxxx
CLAUDE_MODEL=claude-sonnet-4-20250514
GEMINI_MODEL=gemini-2.0-flash
```

---

## Project Structure

### Create the following directory structure:
```bash
motion-analysis-agent/
├── .env                          # API keys (DO NOT COMMIT)
├── .env.example                  # Template for API keys
├── .gitignore                    # Git ignore file
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── config/
│   ├── __init__.py
│   ├── settings.py              # Configuration settings
│   └── exercise_rules.json      # Biomechanics rules
├── data/
│   ├── uploads/                 # User uploaded videos
│   ├── frames/                  # Extracted frames
│   ├── clips/                   # Extracted video clips
│   └── database/                # ChromaDB storage
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── frame_extractor.py   # FFmpeg frame extraction
│   │   ├── pose_estimator.py    # MediaPipe pose detection
│   │   └── clip_embedder.py     # CLIP embedding generation
│   ├── storage/
│   │   ├── __init__.py
│   │   └── vector_store.py      # ChromaDB operations
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── query_parser.py      # Query understanding agent
│   │   ├── retrieval_agent.py   # Frame retrieval agent
│   │   └── analysis_agent.py    # Video analysis agent
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── semantic_search.py   # CLIP search tool
│   │   ├── clip_extractor.py    # Video clip extraction
│   │   ├── video_analyzer.py    # Gemini analysis tool
│   │   └── rule_validator.py    # Biomechanics validator
│   └── utils/
│       ├── __init__.py
│       ├── video_utils.py       # Video helper functions
│       └── angle_calculator.py  # Joint angle calculations
├── notebooks/
│   └── demo.ipynb               # Jupyter notebook for demos
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_agents.py
│   └── test_tools.py
└── main.py                      # Main application entry point
```

### Create the structure:
```bash
# Create all directories
mkdir -p config data/uploads data/frames data/clips data/database
mkdir -p src/preprocessing src/storage src/agents src/tools src/utils
mkdir -p notebooks tests

# Create __init__.py files
touch config/__init__.py
touch src/__init__.py
touch src/preprocessing/__init__.py
touch src/storage/__init__.py
touch src/agents/__init__.py
touch src/tools/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py
```

---

## Configuration

### Step 1: Create settings.py

Create `config/settings.py`:
```python
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
FRAMES_DIR = DATA_DIR / "frames"
CLIPS_DIR = DATA_DIR / "clips"
DATABASE_DIR = DATA_DIR / "database"

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model configurations
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Video processing settings
FRAME_EXTRACTION_FPS = 1  # Extract 1 frame per second
VIDEO_CLIP_BUFFER = 2  # Add 2 seconds buffer to clips

# CLIP settings
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# ChromaDB settings
CHROMADB_COLLECTION_NAME = "video_frames"

# Analysis settings
TOP_K_RESULTS = 20  # Number of frames to retrieve
TEMPORAL_CLUSTER_THRESHOLD = 5  # Seconds between clips
```

---

### Step 2: Create exercise_rules.json

Create `config/exercise_rules.json`:
```json
{
  "squat": {
    "name": "Squat",
    "description": "Basic squat form analysis",
    "rules": [
      {
        "rule_id": "squat_knee_angle",
        "name": "Knee Angle Check",
        "description": "Knees should not bend past 90 degrees",
        "severity": "high",
        "check": {
          "joint": "knee",
          "angle": {
            "min": 90,
            "max": 180
          }
        },
        "violation_message": "Knees bent too deep ({{value}}°). Risk of knee injury. Keep knees at or above 90°."
      },
      {
        "rule_id": "squat_back_angle",
        "name": "Back Alignment",
        "description": "Back should remain straight",
        "severity": "medium",
        "check": {
          "joint": "spine",
          "angle": {
            "min": 160,
            "max": 190
          }
        },
        "violation_message": "Back rounding detected ({{value}}°). Risk of spine injury. Maintain straight back (>160°)."
      },
      {
        "rule_id": "squat_knee_tracking",
        "name": "Knee Tracking",
        "description": "Knees should track over toes",
        "severity": "high",
        "check": {
          "type": "alignment",
          "joints": ["knee", "ankle"]
        },
        "violation_message": "Knee valgus detected. Knees caving inward. Push knees outward."
      }
    ]
  },
  "pushup": {
    "name": "Push-up",
    "description": "Push-up form analysis",
    "rules": [
      {
        "rule_id": "pushup_elbow_angle",
        "name": "Elbow Angle",
        "description": "Elbows should reach 80-90 degrees at bottom",
        "severity": "medium",
        "check": {
          "joint": "elbow",
          "angle": {
            "min": 80,
            "max": 180
          }
        },
        "violation_message": "Incomplete range of motion ({{value}}°). Lower chest closer to ground."
      },
      {
        "rule_id": "pushup_body_alignment",
        "name": "Body Alignment",
        "description": "Body should form straight line",
        "severity": "high",
        "check": {
          "type": "alignment",
          "joints": ["shoulder", "hip", "ankle"]
        },
        "violation_message": "Body sagging or hips raised. Maintain straight line from head to heels."
      }
    ]
  },
  "downward_dog": {
    "name": "Downward Dog",
    "description": "Yoga downward dog pose analysis",
    "rules": [
      {
        "rule_id": "dd_back_straight",
        "name": "Spine Alignment",
        "description": "Spine should be straight",
        "severity": "medium",
        "check": {
          "joint": "spine",
          "angle": {
            "min": 170,
            "max": 190
          }
        },
        "violation_message": "Spine curved ({{value}}°). Lengthen through the spine."
      }
    ]
  }
}
```

---

### Step 3: Create .gitignore

Create `.gitignore`:
```
# Environment variables
.env

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Data files
data/uploads/*
data/frames/*
data/clips/*
data/database/*

# Keep directory structure
!data/uploads/.gitkeep
!data/frames/.gitkeep
!data/clips/.gitkeep
!data/database/.gitkeep

# Jupyter
.ipynb_checkpoints
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/
```

---

### Step 4: Create placeholder files
```bash
# Create .gitkeep files to preserve directory structure
touch data/uploads/.gitkeep
touch data/frames/.gitkeep
touch data/clips/.gitkeep
touch data/database/.gitkeep
```

---

## Testing the Setup

### Step 1: Create test_setup.py

Create `test_setup.py` in project root:
```python
#!/usr/bin/env python3
"""
Setup verification script
Checks if all dependencies and configurations are correct
"""

import sys
from pathlib import Path

def test_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print("✓ Python version OK:", f"{version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print("✗ Python version too old. Need 3.9+, have:", f"{version.major}.{version.minor}")
        return False

def test_ffmpeg():
    """Check FFmpeg installation"""
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print("✓ FFmpeg installed:", version.split(' ')[2])
            return True
        else:
            print("✗ FFmpeg not working")
            return False
    except FileNotFoundError:
        print("✗ FFmpeg not found in PATH")
        return False

def test_imports():
    """Test critical package imports"""
    packages = {
        'cv2': 'OpenCV',
        'mediapipe': 'MediaPipe',
        'transformers': 'Transformers',
        'torch': 'PyTorch',
        'chromadb': 'ChromaDB',
        'langchain': 'LangChain',
        'anthropic': 'Anthropic',
        'google.generativeai': 'Google GenAI'
    }
    
    all_ok = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name} imported successfully")
        except ImportError as e:
            print(f"✗ {name} import failed:", str(e))
            all_ok = False
    
    return all_ok

def test_api_keys():
    """Check if API keys are set"""
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    keys = {
        'ANTHROPIC_API_KEY': 'Claude API',
        'GOOGLE_API_KEY': 'Gemini API'
    }
    
    all_ok = True
    for key, name in keys.items():
        value = os.getenv(key)
        if value and value.startswith('sk-') or value.startswith('AIza'):
            print(f"✓ {name} key found")
        else:
            print(f"✗ {name} key not set or invalid")
            all_ok = False
    
    return all_ok

def test_directories():
    """Check if required directories exist"""
    dirs = [
        'data/uploads',
        'data/frames',
        'data/clips',
        'data/database',
        'config',
        'src'
    ]
    
    all_ok = True
    for dir_path in dirs:
        if Path(dir_path).exists():
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Directory missing: {dir_path}")
            all_ok = False
    
    return all_ok

def test_clip_model():
    """Test CLIP model loading"""
    try:
        from transformers import CLIPModel, CLIPProcessor
        print("Loading CLIP model (may take a minute)...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("✓ CLIP model loaded successfully")
        return True
    except Exception as e:
        print("✗ CLIP model loading failed:", str(e))
        return False

def test_mediapipe():
    """Test MediaPipe pose model"""
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        print("✓ MediaPipe Pose initialized successfully")
        pose.close()
        return True
    except Exception as e:
        print("✗ MediaPipe initialization failed:", str(e))
        return False

def main():
    print("=" * 60)
    print("MOTION ANALYSIS AGENT - SETUP VERIFICATION")
    print("=" * 60)
    print()
    
    tests = [
        ("Python Version", test_python_version),
        ("FFmpeg Installation", test_ffmpeg),
        ("Package Imports", test_imports),
        ("API Keys", test_api_keys),
        ("Directory Structure", test_directories),
        ("CLIP Model", test_clip_model),
        ("MediaPipe Pose", test_mediapipe)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- Testing {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Setup is complete.")
        print("\nNext steps:")
        print("1. Add a test video to data/uploads/")
        print("2. Run: python main.py --preprocess data/uploads/video.mp4")
        print("3. Run: python main.py --query 'Is my squat safe?'")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

### Step 2: Run Setup Verification
```bash
# Make script executable (Linux/macOS)
chmod +x test_setup.py

# Run verification
python test_setup.py
```

**Expected output:**
```
============================================================
MOTION ANALYSIS AGENT - SETUP VERIFICATION
============================================================

--- Testing Python Version ---
✓ Python version OK: 3.11.5

--- Testing FFmpeg Installation ---
✓ FFmpeg installed: 4.4.2

--- Testing Package Imports ---
✓ OpenCV imported successfully
✓ MediaPipe imported successfully
✓ Transformers imported successfully
✓ PyTorch imported successfully
✓ ChromaDB imported successfully
✓ LangChain imported successfully
✓ Anthropic imported successfully
✓ Google GenAI imported successfully

--- Testing API Keys ---
✓ Claude API key found
✓ Gemini API key found

--- Testing Directory Structure ---
✓ Directory exists: data/uploads
✓ Directory exists: data/frames
✓ Directory exists: data/clips
✓ Directory exists: data/database
✓ Directory exists: config
✓ Directory exists: src

--- Testing CLIP Model ---
Loading CLIP model (may take a minute)...
✓ CLIP model loaded successfully

--- Testing MediaPipe Pose ---
✓ MediaPipe Pose initialized successfully

============================================================
SUMMARY
============================================================
✓ Python Version: PASS
✓ FFmpeg Installation: PASS
✓ Package Imports: PASS
✓ API Keys: PASS
✓ Directory Structure: PASS
✓ CLIP Model: PASS
✓ MediaPipe Pose: PASS

Total: 7/7 tests passed

🎉 All tests passed! Setup is complete.

Next steps:
1. Add a test video to data/uploads/
2. Run: python main.py --preprocess data/uploads/video.mp4
3. Run: python main.py --query 'Is my squat safe?'
```

---

## Troubleshooting

### Issue 1: FFmpeg not found

**Error**: `ffmpeg: command not found`

**Solution**:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
# Add to PATH environment variable
```

---

### Issue 2: MediaPipe installation fails

**Error**: `ERROR: Failed building wheel for mediapipe`

**Solution**:
```bash
# Install system dependencies first
# Ubuntu/Debian:
sudo apt install python3-dev libgl1-mesa-glx libglib2.0-0

# Then retry
pip install mediapipe
```

---

### Issue 3: PyTorch CUDA issues

**Error**: `CUDA not available`

**Solution**:
```bash
# If you have NVIDIA GPU, install CUDA version
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# If no GPU, CPU version is fine
pip install torch torchvision torchaudio
```

---

### Issue 4: ChromaDB SQLite error

**Error**: `SQLite version too old`

**Solution**:
```bash
# Install newer SQLite
pip install pysqlite3-binary

# Update ChromaDB
pip install --upgrade chromadb
```

---

### Issue 5: API Key not recognized

**Error**: `API key not found`

**Solution**:
1. Check `.env` file exists in project root
2. Verify no spaces around `=` in `.env`:
```
   ANTHROPIC_API_KEY=sk-ant-xxxxx  ✓ Correct
   ANTHROPIC_API_KEY = sk-ant-xxxxx  ✗ Wrong
```
3. Restart terminal/IDE after adding keys
4. Test with:
```python
   from dotenv import load_dotenv
   import os
   load_dotenv()
   print(os.getenv("ANTHROPIC_API_KEY"))
```

---

### Issue 6: Permission denied errors

**Error**: `Permission denied: /data/uploads`

**Solution**:
```bash
# Fix permissions
chmod -R 755 data/
```

---

### Issue 7: Out of memory errors

**Error**: `RuntimeError: out of memory`

**Solution**:
1. Reduce frame extraction rate in `config/settings.py`:
```python
   FRAME_EXTRACTION_FPS = 0.5  # Extract 1 frame every 2 seconds
```
2. Process shorter videos first
3. Close other applications

---

## Next Steps

After successful setup:

1. **Download a test video**:
```bash
   # Example: download a squat tutorial video
   # Place in data/uploads/test_squat.mp4
```

2. **Run preprocessing** (covered in next documentation)

3. **Test a query** (covered in next documentation)

4. **Read the API documentation** for advanced usage

---

## Getting Help

If you encounter issues not covered here:

1. Check the logs in `logs/` directory
2. Verify all tests pass: `python test_setup.py`
3. Check API key validity at provider consoles
4. Review error messages carefully

---

## Useful Commands Reference
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Update all packages
pip install --upgrade -r requirements.txt

# Check installed packages
pip list

# Run setup verification
python test_setup.py

# Clear cache (if having import issues)
find . -type d -name __pycache__ -exec rm -r {} +  # Linux/macOS
```

---

**Setup complete! You're ready to start building the motion analysis agent.**