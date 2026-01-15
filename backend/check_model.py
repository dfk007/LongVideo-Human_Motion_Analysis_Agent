
import vertexai
from vertexai.generative_models import GenerativeModel
import os
import sys

project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "mycompanion-daudfarzand89-4jan")
location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
model_name = os.getenv("VERTEX_AI_MODEL", "gemini-2.5-flash")

print(f"Testing Model: {model_name}")

try:
    vertexai.init(project=project_id, location=location)
    model = GenerativeModel(model_name)
    response = model.generate_content("Hello, are you there?")
    print(f"SUCCESS: {model_name} responded: {response.text}")
except Exception as e:
    print(f"FAILED: {model_name} error: {e}")
