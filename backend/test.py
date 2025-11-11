import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# List all models and their capabilities
for model in genai.list_models():
    print(model.name, model.supported_generation_methods)
