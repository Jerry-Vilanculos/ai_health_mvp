import google.generativeai as genai
import os

# Make sure your GEMINI_API_KEY is set
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

for m in genai.list_models():
    print(m.name)
