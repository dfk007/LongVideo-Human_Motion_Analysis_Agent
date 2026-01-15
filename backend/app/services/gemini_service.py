import os
import google.generativeai as genai
from app.core.config import settings
import time

# Configure GenAI
genai.configure(api_key=os.getenv("VERTEX_API_KEY"))

class GeminiService:
    def __init__(self):
        self.model_name = settings.GEMINI_MODEL_NAME
        self.model = genai.GenerativeModel(self.model_name)

    def upload_video(self, video_path: str):
        """Uploads video to Google AI Studio File API"""
        print(f"Uploading file: {video_path}")
        video_file = genai.upload_file(path=video_path)
        
        # Wait for processing
        while video_file.state.name == "PROCESSING":
            print("Processing video...")
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
            
        if video_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed: {video_file.state.name}")
            
        print(f"Video ready: {video_file.name}")
        return video_file

    def describe_segment(self, video_path: str) -> str:
        """
        Uploads a video segment and gets a detailed description.
        """
        try:
            video_file = self.upload_video(video_path)
            
            prompt = """
            Analyze this video segment of human physical activity.
            Provide a detailed, objective description of:
            1. The specific movement or exercise being performed.
            2. Key body mechanics (posture, limb angles, speed).
            3. Any notable errors, safety concerns, or good technique indicators.
            4. The environment and equipment used.
            
            Be concise but thorough. Focus on visual evidence.
            """
            
            response = self.model.generate_content([video_file, prompt])
            
            # Clean up file after analysis to save storage/limit
            # genai.delete_file(video_file.name) 
            # (Optional: keep it if we want to re-query, but for now let's assume one-pass indexing)
            
            return response.text
        except Exception as e:
            print(f"Error describing segment: {e}")
            return "Error analyzing segment."

gemini_service = GeminiService()
