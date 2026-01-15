import os
import json
import logging
from pathlib import Path
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
        
        # Load biomechanics guidelines
        self.guidelines = self._load_guidelines()
        logger.info(f"Loaded biomechanics guidelines for {len(self.guidelines.get('exercises', {}))} exercises")
    
    def _load_guidelines(self) -> dict:
        """Load biomechanics guidelines from JSON file"""
        try:
            # Get the directory where this file (gemini_service.py) is located
            current_dir = Path(__file__).parent
            guidelines_path = current_dir / "biomechanics_guidelines.json"
            
            if not guidelines_path.exists():
                logger.warning(f"Guidelines file not found at {guidelines_path}. Using empty guidelines.")
                return {"exercises": {}}
            
            with open(guidelines_path, 'r') as f:
                guidelines = json.load(f)
            
            logger.info(f"Successfully loaded guidelines from {guidelines_path}")
            return guidelines
        except Exception as e:
            logger.error(f"Failed to load guidelines: {e}")
            return {"exercises": {}}
    
    def _build_grounded_prompt(self, base_prompt: str, exercise_type: str = None) -> str:
        """
        Build a prompt grounded in biomechanics guidelines
        
        Args:
            base_prompt: The original prompt/question
            exercise_type: Type of exercise (squat, deadlift, etc.) - if None, includes all
        
        Returns:
            Enhanced prompt with guidelines context
        """
        if not self.guidelines or not self.guidelines.get('exercises'):
            return base_prompt
        
        # Build guidelines context
        guidelines_text = "# BIOMECHANICS SAFETY GUIDELINES\n\n"
        guidelines_text += "You are a certified biomechanics expert. ALWAYS base your analysis on these evidence-based standards:\n\n"
        
        if exercise_type and exercise_type.lower() in self.guidelines['exercises']:
            # Include only specific exercise guidelines
            exercise_data = self.guidelines['exercises'][exercise_type.lower()]
            guidelines_text += self._format_exercise_guidelines(exercise_type.lower(), exercise_data)
        else:
            # Include all exercise guidelines
            for ex_name, ex_data in self.guidelines['exercises'].items():
                guidelines_text += self._format_exercise_guidelines(ex_name, ex_data)
                guidelines_text += "\n---\n\n"
        
        guidelines_text += "\n## CRITICAL INSTRUCTIONS:\n"
        guidelines_text += "1. ALWAYS cite specific standards when identifying issues (e.g., 'Knee angle: 82° - Standard requires ≥90°')\n"
        guidelines_text += "2. Reference the SOURCE for each standard you cite\n"
        guidelines_text += "3. If guidelines don't cover the question, clearly state 'No established standard available'\n"
        guidelines_text += "4. Provide severity levels: CRITICAL, HIGH, MEDIUM, LOW based on injury risk\n"
        guidelines_text += "5. Always give corrective recommendations from the guidelines\n\n"
        
        # Combine guidelines with original prompt
        full_prompt = f"{guidelines_text}\n## USER QUERY:\n{base_prompt}"
        
        return full_prompt
    
    def _format_exercise_guidelines(self, exercise_name: str, exercise_data: dict) -> str:
        """Format exercise guidelines into readable text for the prompt"""
        text = f"## {exercise_data.get('name', exercise_name.upper())}\n"
        text += f"{exercise_data.get('description', '')}\n\n"
        
        # Safety Standards
        text += "### Safety Standards:\n"
        for standard_name, standard_data in exercise_data.get('safety_standards', {}).items():
            text += f"\n**{standard_name.replace('_', ' ').title()}:**\n"
            
            if 'safe_range' in standard_data:
                text += f"- Safe Range: {standard_data['safe_range'][0]}-{standard_data['safe_range'][1]} {standard_data.get('unit', '')}\n"
            if 'optimal_range' in standard_data:
                text += f"- Optimal Range: {standard_data['optimal_range'][0]}-{standard_data['optimal_range'][1]} {standard_data.get('unit', '')}\n"
            if 'standard' in standard_data:
                text += f"- Standard: {standard_data['standard']}\n"
            if 'measurement' in standard_data:
                text += f"- Measurement: {standard_data['measurement']}\n"
            if 'violation_risk' in standard_data:
                text += f"- Risk Level: {standard_data['violation_risk'].upper()}\n"
            if 'source' in standard_data:
                text += f"- Source: {standard_data['source']}\n"
        
        # Common Errors
        if 'common_errors' in exercise_data:
            text += "\n### Common Errors:\n"
            for error in exercise_data['common_errors']:
                text += f"\n**{error['error']}:**\n"
                text += f"- Detection: {error.get('detection', 'N/A')}\n"
                text += f"- Risk: {error.get('risk', 'N/A')}\n"
                text += f"- Correction: {error.get('correction', 'N/A')}\n"
                if 'source' in error:
                    text += f"- Source: {error['source']}\n"
        
        return text
    
    def describe_segment(self, video_path: str, exercise_type: str = None) -> str:
        """
        Analyze video segment with biomechanics guidelines grounding
        
        Args:
            video_path: Path to video file
            exercise_type: Type of exercise being performed (optional)
        """
        try:
            base_prompt = """Analyze this video segment for movement quality, safety, and biomechanics.

Provide your analysis in this format:
1. MOVEMENT IDENTIFIED: [What exercise/movement is being performed]
2. SAFETY ASSESSMENT: [Overall safety rating and specific concerns with measurements]
3. FORM VIOLATIONS: [List any violations with cited standards]
4. RECOMMENDATIONS: [Specific corrections based on guidelines]

Be specific with measurements and always cite the relevant safety standards."""

            # Ground the prompt in guidelines
            grounded_prompt = self._build_grounded_prompt(base_prompt, exercise_type)
            
            with open(video_path, "rb") as f:
                video_bytes = f.read()
            
            video_part = Part.from_data(data=video_bytes, mime_type="video/mp4")
            response = self.model.generate_content([video_part, grounded_prompt])
            return response.text
        except Exception as e:
            logger.error(f"Error describing segment: {e}")
            return f"Error analyzing segment: {str(e)}"
    
    def generate_answer(self, prompt: str, exercise_type: str = None, context: dict = None) -> str:
        """
        Generate answer grounded in biomechanics guidelines
        
        Args:
            prompt: User question
            exercise_type: Type of exercise being discussed (optional)
            context: Additional context like pose data, timestamps, etc.
        """
        try:
            # Build enhanced prompt with context
            enhanced_prompt = prompt
            
            if context:
                enhanced_prompt += "\n\n## CONTEXT:\n"
                if 'pose_data' in context:
                    enhanced_prompt += f"Pose measurements: {json.dumps(context['pose_data'], indent=2)}\n"
                if 'timestamp' in context:
                    enhanced_prompt += f"Timestamp: {context['timestamp']}s\n"
                if 'video_segment' in context:
                    enhanced_prompt += f"Video segment: {context['video_segment']}\n"
            
            # Ground in guidelines
            grounded_prompt = self._build_grounded_prompt(enhanced_prompt, exercise_type)
            
            response = self.model.generate_content(grounded_prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I encountered an error generating the answer."
    
    def detect_exercise_type(self, description: str) -> str:
        """
        Detect exercise type from movement description
        Returns the exercise type key for guidelines lookup
        """
        description_lower = description.lower()
        
        # Simple keyword matching - can be enhanced with ML
        exercise_keywords = {
            'squat': ['squat', 'squatting'],
            'deadlift': ['deadlift', 'deadlifting'],
            'pushup': ['pushup', 'push-up', 'push up'],
            'pullup': ['pullup', 'pull-up', 'pull up'],
            'benchpress': ['bench press', 'bench pressing'],
            'row': ['row', 'rowing'],
            'lunge': ['lunge', 'lunging'],
            'plank': ['plank', 'planking']
        }
        
        for exercise, keywords in exercise_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                return exercise
        
        return None

gemini_service = GeminiService()