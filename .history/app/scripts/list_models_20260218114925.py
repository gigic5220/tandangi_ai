import os
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

for m in client.models.list():
    # generateContent 가능한 모델만 보고 싶으면 아래처럼 필터링
    methods = getattr(m, "supported_generation_methods", None) or getattr(m, "supportedGenerationMethods", [])
    if "generateContent" in methods:
        print(m.name)
