import os
import json
import logging
import google.generativeai as genai
from pathlib import Path
from app.core.config import settings
import time

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        # Configure Google AI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY not found. Gemini service may fail.")
        
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL or "gemini-1.5-flash")
        
        # Load biomechanics guidelines
        self.guidelines = self._load_guidelines()
    
    def _load_guidelines(self) -> dict:
        try:
            current_dir = Path(__file__).parent
            guidelines_path = current_dir / "biomechanics_guidelines.json"
            
            if not guidelines_path.exists():
                return {"exercises": {}}
            
            with open(guidelines_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {"exercises": {}}
    
    def describe_segment(self, video_path: str) -> str:
        """
        Analyze video segment using Gemini (Multimodal)
        """
        if not os.path.exists(video_path):
            return "Error: Video file not found"

        try:
            # Upload file to Gemini File API
            # This is more robust than base64 for videos
            logger.info(f"Uploading {video_path} to Gemini...")
            video_file = genai.upload_file(path=video_path)
            
            # Wait for processing state to be ACTIVE
            while video_file.state.name == "PROCESSING":
                time.sleep(1)
                video_file = genai.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                raise ValueError(f"Video processing failed: {video_file.state.name}")

            prompt = """Analyze this video segment for human motion analysis.
            1. Describe the movement/exercise being performed in detail.
            2. Note any obvious form errors or safety risks.
            3. Provide a summary suitable for retrieval.
            """

            response = self.model.generate_content([video_file, prompt])
            
            # Clean up file after use (optional but good practice)
            # genai.delete_file(video_file.name)
            
            return response.text
        except Exception as e:
            logger.error(f"Error describing segment: {e}")
            return f"Error analyzing segment: {str(e)}"

gemini_service = GeminiService()
