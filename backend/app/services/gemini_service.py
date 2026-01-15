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
        self.use_vertex = True # Enforce Vertex AI usage with Service Account
        
        if not HAS_VERTEX:
            raise RuntimeError("vertexai library not installed. Add google-cloud-aiplatform to requirements.")
        
        try:
            # Vertex AI automatically picks up GOOGLE_APPLICATION_CREDENTIALS
            vertexai.init(
                project=settings.GOOGLE_CLOUD_PROJECT,
                location=settings.GOOGLE_CLOUD_LOCATION
            )
            self.model = GenerativeModel(settings.GEMINI_MODEL_NAME)
            logger.info(f"Vertex AI initialized with Service Account. Model: {settings.GEMINI_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            raise

    def describe_segment(self, video_path: str) -> str:
        """
        Uploads a video segment and gets a detailed description.
        """
        prompt = """
        Analyze this video segment of human physical activity.
        Provide a detailed, objective description of:
        1. The specific movement or exercise being performed.
        2. Key body mechanics (posture, limb angles, speed).
        3. Any notable errors, safety concerns, or good technique indicators.
        4. The environment and equipment used.
        
        Be concise but thorough. Focus on visual evidence.
        """

        try:
            # Vertex AI: Send video bytes directly
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
