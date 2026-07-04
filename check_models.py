import os
import requests
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def test_groq():
    print("1. Testing Groq...", end=" ")
    try:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))
        # Llama 3.1 is the LTS release for 2025
        client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1
        )
        print("✅ Success (llama-3.1-8b-instant)")
    except Exception as e:
        print(f"❌ Failed: {e}")

def test_openrouter():
    print("2. Testing OpenRouter...", end=" ")
    try:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
        # Trying the safest free model for Nov 2025
        model = "meta-llama/llama-3.2-11b-vision-instruct:free"
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            extra_body={"headers": {"HTTP-Referer": "http://localhost"}}
        )
        print(f"✅ Success ({model})")
    except Exception as e:
        print(f"❌ Failed: {e}")

def test_gemini():
    print("3. Testing Gemini...", end=" ")
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        # gemini-1.5 is dead (Sept 2025). Using 2.5.
        model = genai.GenerativeModel("gemini-2.5-flash")
        model.generate_content("hi")
        print("✅ Success (gemini-2.5-flash)")
    except Exception as e:
        print(f"❌ Failed: {e}")

def test_apifreellm():
    print("4. Testing ApiFreeLLM...", end=" ")
    try:
        resp = requests.post("https://apifreellm.com/api/chat", json={"message": "hi"}, timeout=5)
        if resp.ok: print("✅ Success")
        else: print(f"❌ Failed (Status {resp.status_code})")
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    test_groq()
    test_openrouter()
    test_gemini()
    test_apifreellm()