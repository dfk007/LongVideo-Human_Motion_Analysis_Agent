import os
import logging
from app.core.config import settings

# Vertex AI imports
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Part
    HAS_VERTEX = True
except ImportError:
    HAS_VERTEX = False

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        self.use_vertex = True 
        
        if not HAS_VERTEX:
            raise RuntimeError("vertexai library not installed.")
        
        try:
            vertexai.init(
                project=settings.GOOGLE_CLOUD_PROJECT,
                location=settings.GOOGLE_CLOUD_LOCATION
            )
            # Using the new model name VERTEX_AI_MODEL
            self.model = GenerativeModel(settings.VERTEX_AI_MODEL)
            logger.info(f"Vertex AI initialized. Model: {settings.VERTEX_AI_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            raise

    def describe_segment(self, video_path: str) -> str:
        try:
            prompt = "Analyze this video segment and describe movement, safety, and mechanics."
            with open(video_path, "rb") as f:
                video_bytes = f.read()
            
            video_part = Part.from_data(data=video_bytes, mime_type="video/mp4")
            response = self.model.generate_content([video_part, prompt])
            return response.text
        except Exception as e:
            logger.error(f"Error describing segment: {e}")
            return f"Error analyzing segment: {str(e)}"

    def generate_answer(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I encountered an error generating the answer."

gemini_service = GeminiService()