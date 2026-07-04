import os
import json
import requests
import aiohttp
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. CLIENT CONFIGURATION & INITIALIZATION ---

# Groq Client 1 (Handles: Llama 3.3, Kimi K2, ALLaM 7B)
GROQ_CLIENT_1 = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY_1")
)

# Groq Client 2 (Handles: Qwen 3, Compound, GPT-OSS 120B)
GROQ_CLIENT_2 = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY_2")
)

# SambaNova Client (Handles: Llama 3.3 70B)
SAMBANOVA_CLIENT = AsyncOpenAI(
    base_url="https://api.sambanova.ai/v1",
    api_key=os.getenv("SAMBANOVA_API_KEY")
)

# Google Gemini Client (Handles: Flash 2.0)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Cloudflare Configuration (Handles: Llama 3.2)
CF_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CF_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
CF_HEADERS = {"Authorization": f"Bearer {CF_API_TOKEN}"}

# --- HELPER FUNCTIONS ---

def build_memory_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Manually constructs a conversation script from the message history.
    This is required for stateless APIs (Pollinations, ApiFreeLLM, Cloudflare REST)
    that do not accept a list of message objects.
    """
    prompt = "Conversation History:\n"
    for m in messages:
        role = "User" if m['role'] == 'user' else "AI"
        prompt += f"{role}: {m['content']}\n"
    prompt += "AI:"
    return prompt

# --- 2. GENERATOR FUNCTIONS (STREAMING LOGIC) ---

async def clean_think_filter(generator):
    """Strip out any <think>...</think> blocks from the streaming response"""
    in_think = False
    buffer = ""
    
    async for chunk in generator:
        if in_think:
            buffer += chunk
            if "</think>" in buffer:
                idx = buffer.find("</think>")
                remaining = buffer[idx + 8:]
                buffer = ""
                in_think = False
                if remaining:
                    yield remaining
            continue
            
        buffer += chunk
        if "<think>" in buffer:
            in_think = True
            idx = buffer.find("<think>")
            before = buffer[:idx]
            buffer = buffer[idx:]
            if before:
                yield before
        else:
            # Check for partial matches of `<think>`
            prefix_of_think = False
            for i in range(1, len(buffer) + 1):
                part = buffer[:i]
                if "<think>".startswith(part):
                    prefix_of_think = True
                    
            if prefix_of_think:
                if len(buffer) > 10:
                    yield buffer
                    buffer = ""
            else:
                yield buffer
                buffer = ""
                
    if not in_think and buffer:
        yield buffer

# === GROQ ENGINE 1 MODELS ===

async def generate_groq_llama(messages):
    """Stream Llama 3.3 70B Versatile"""
    try:
        stream = await GROQ_CLIENT_1.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"Groq Llama Error: {str(e)}"

async def generate_groq_qwen36(messages):
    """Stream Qwen 3.6 27B (Cleaned of think blocks)"""
    async def raw_generator():
        try:
            stream = await GROQ_CLIENT_1.chat.completions.create(
                model="qwen/qwen3.6-27b",
                messages=messages,
                stream=True
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"Groq Qwen 3.6 Error: {str(e)}"
            
    async for chunk in clean_think_filter(raw_generator()):
        yield chunk

async def generate_groq_allam(messages):
    """Stream ALLaM 2 7B (Saudi AI via Llama 3.1 8B Instant)"""
    try:
        stream = await GROQ_CLIENT_1.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"Groq ALLaM Error: {str(e)}"

# === GROQ ENGINE 2 MODELS ===

async def generate_groq_qwen(messages):
    """Stream Qwen 3 32B (Cleaned of think blocks)"""
    async def raw_generator():
        try:
            stream = await GROQ_CLIENT_2.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=messages,
                stream=True
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"Groq Qwen Error: {str(e)}"
            
    async for chunk in clean_think_filter(raw_generator()):
        yield chunk

async def generate_groq_compound(messages):
    """Stream Groq Compound (Agentic System)"""
    try:
        stream = await GROQ_CLIENT_2.chat.completions.create(
            model="groq/compound",
            messages=messages,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"Groq Compound Error: {str(e)}"

async def generate_groq_gptoss(messages):
    """Stream OpenAI GPT-OSS 120B"""
    try:
        stream = await GROQ_CLIENT_2.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"Groq GPT-OSS Error: {str(e)}"

# === OTHER PROVIDERS ===

async def generate_groq_scout(messages):
    """Stream Llama 4 Scout 17B"""
    try:
        stream = await GROQ_CLIENT_2.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"Groq Llama 4 Scout Error: {str(e)}"

async def generate_gemini(messages, image=None):
    """Stream Gemini (Auto-switching versions, supporting multimodal vision)"""
    # Convert standard messages format to Gemini's history format
    gemini_history = []
    for m in messages[:-1]:
        role = "user" if m["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [m["content"]]})
    
    current_message = messages[-1]["content"]

    import base64
    image_part = None
    if image and "data" in image and "mime_type" in image:
        try:
            image_data = base64.b64decode(image["data"])
            image_part = {
                "mime_type": image["mime_type"],
                "data": image_data
            }
        except Exception:
            pass

    # Priority list: Experimental -> New Stable -> Old Stable
    models_to_try = ["gemini-2.0-flash-exp", "gemini-2.5-flash", "gemini-1.5-flash"]
    
    for model_id in models_to_try:
        try:
            model = genai.GenerativeModel(model_id)
            if image_part:
                # Compile full contents list for direct generation (since history with mixed types can be strict)
                contents = []
                for h in gemini_history:
                    contents.append({"role": h["role"], "parts": h["parts"]})
                # Add final user message + image part
                contents.append({"role": "user", "parts": [current_message, image_part]})
                
                response = await model.generate_content_async(contents, stream=True)
            else:
                chat = model.start_chat(history=gemini_history)
                response = await chat.send_message_async(current_message, stream=True)
                
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
            return # Exit function on success
        except Exception:
            continue # Try next model in list
            
    # Fallback: All Gemini models failed, run Llama 3.3 on Groq
    async for chunk in generate_groq_llama(messages):
        yield chunk

async def generate_pollinations(messages):
    """Stream Pollinations (GPT-4o Proxy) - Stateless"""
    full_text = build_memory_prompt(messages)
    encoded_prompt = requests.utils.quote(full_text)
    # Using the 'openai' model which maps to GPT-4o
    url = f"https://text.pollinations.ai/{encoded_prompt}?model=openai"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    async for chunk in response.content.iter_chunked(1024):
                        if chunk:
                            yield chunk.decode('utf-8')
                else:
                    yield f"Pollinations Error: HTTP {response.status}"
    except Exception as e:
        yield f"Pollinations Connection Error: {str(e)}"

async def generate_cloudflare(messages):
    """Stream Cloudflare Workers AI (Llama 3.2) - Stateless"""
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/@cf/meta/llama-3.2-3b-instruct"
    full_text = build_memory_prompt(messages)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, 
                headers=CF_HEADERS, 
                json={"prompt": full_text, "stream": True}
            ) as response:
                async for line in response.content:
                    line_text = line.decode("utf-8").strip()
                    if line_text.startswith("data: "):
                        try:
                            # Cloudflare streams data: {"response": "word"}
                            data = json.loads(line_text[6:])
                            if "response" in data:
                                yield data["response"]
                        except:
                            pass
    except Exception as e:
        yield f"Cloudflare Error: {str(e)}"

async def generate_free(messages):
    """Fetch ApiFreeLLM (Synchronous) - Stateless"""
    url = "https://apifreellm.com/api/chat"
    full_text = build_memory_prompt(messages)
    
    try:
        # Using sync requests with timeout, yielded as a single chunk
        # The frontend handles the "typing" animation for this big chunk
        resp = requests.post(url, json={"message": full_text}, timeout=60)
        if resp.ok:
            json_response = resp.json()
            # Check for common response keys
            text = json_response.get("response") or json_response.get("message") or str(json_response)
            yield text
        else:
            yield f"ApiFreeLLM Error: Status {resp.status_code}"
    except Exception as e:
        yield f"ApiFreeLLM Connection Error: {str(e)}"

# --- 3. ROUTE HANDLERS ---

@app.get("/")
async def get_ui():
    """Serves the main HTML interface"""
    with open("index.html", "r", encoding='utf-8') as f:
        return HTMLResponse(content=f.read())

@app.get("/meeting.png")
async def get_meeting_logo():
    """Serves the meeting logo image file"""
    return FileResponse("meeting.png")

@app.get("/favicon.ico")
async def get_favicon():
    """Serves the meeting logo as favicon icon"""
    return FileResponse("meeting.png", media_type="image/png")

@app.post("/chat/{provider}")
async def chat_endpoint(provider: str, request: Request):
    """Main routing endpoint for all AI models"""
    try:
        data = await request.json()
        messages = data.get("messages", [])
        image = data.get("image", None)

        # Groq Key 1 Routes
        if provider == "groq_llama":
            return StreamingResponse(generate_groq_llama(messages), media_type="text/plain")
        if provider == "groq_qwen36":
            return StreamingResponse(generate_groq_qwen36(messages), media_type="text/plain")
        if provider == "groq_allam":
            return StreamingResponse(generate_groq_allam(messages), media_type="text/plain")
        
        # Groq Key 2 Routes
        if provider == "groq_qwen":
            return StreamingResponse(generate_groq_qwen(messages), media_type="text/plain")
        if provider == "groq_compound":
            return StreamingResponse(generate_groq_compound(messages), media_type="text/plain")
        if provider == "groq_gptoss":
            return StreamingResponse(generate_groq_gptoss(messages), media_type="text/plain")
        
        # Other Provider Routes
        if provider == "groq_scout":
            if image and "data" in image and "mime_type" in image:
                # Format standard user message content with OpenAI multimodal vision content structure
                messages[-1]["content"] = [
                    {"type": "text", "text": messages[-1]["content"]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image['mime_type']};base64,{image['data']}"
                        }
                    }
                ]
            return StreamingResponse(generate_groq_scout(messages), media_type="text/plain")
        if provider == "gemini":
            return StreamingResponse(generate_gemini(messages, image), media_type="text/plain")
        if provider == "poll":
            return StreamingResponse(generate_pollinations(messages), media_type="text/plain")
        if provider == "cf":
            return StreamingResponse(generate_cloudflare(messages), media_type="text/plain")
        if provider == "free":
            return StreamingResponse(generate_free(messages), media_type="text/plain")
        
        return "Invalid Provider ID"
        
    except Exception as e:
        return f"Server Error: {str(e)}"

# --- 4. DEBATE CHAMBER ENDPOINTS ---

MODEL_GENERATORS = {
    "groq_llama": (generate_groq_llama, "Llama 3.3"),
    "groq_qwen36": (generate_groq_qwen36, "Qwen 3.6"),
    "groq_allam": (generate_groq_allam, "ALLaM 7B"),
    "groq_qwen": (generate_groq_qwen, "Qwen 3"),
    "groq_compound": (generate_groq_compound, "Compound"),
    "groq_gptoss": (generate_groq_gptoss, "GPT-OSS"),
    "groq_scout": (generate_groq_scout, "Llama 4 Scout"),
    "gemini": (generate_gemini, "Gemini 2.0"),
    "poll": (generate_pollinations, "GPT-4o"),
    "cf": (generate_cloudflare, "Llama 3.2"),
}

def build_debate_messages(topic: str, history: List[Dict[str, Any]], current_model_id: str, current_model_name: str, turns_left: int) -> List[Dict[str, str]]:
    system_instruction = (
        f"You are {current_model_name}, a critical and open-minded member of the AI Council.\n"
        f"You are debating the topic: \"{topic}\".\n"
        "Your goal: Select a clear stance on this topic and persistently defend it, build on your previous logic, and directly attack opponent weaknesses.\n"
        f"Debate Progress: There are exactly {turns_left} messages remaining in this entire debate (all models combined) before the Scribe calls a final verdict. budget your arguments to prove your point before the debate ends.\n"
        f"{'This is the absolute FINAL speaker turn. Make your final strike at the opposing arguments and conclude your case.' if turns_left == 1 else ''}\n\n"
        "Rhetorical Directives (MANDATORY FORMAT):\n"
        "1. You MUST follow this exact structure (including double line breaks and bullets):\n"
        "**STANCE**: [Your stance or opinion in 3-5 words]\n\n"
        "**REASONING**:\n"
        "* [First key premise or direct refutation targeting opponents' logic]\n"
        "* [Supporting fact or logical deduction]\n\n"
        "2. Strict Word Limit: Stay under 45 words total. Be extremely concise. Avoid all preambles, greetings, intros, or filler text. Start directly with '**STANCE**'.\n"
        "Ideological Directives:\n"
        "3. Reject generic 'woke' platitudes, sugarcoating, or modern sanitized corporate speech. Speak with core realistic, scientific, factual, and critical philosophical rigor.\n"
        "4. Base your arguments on facts, logic, and critical evaluation, even if the conclusion is controversial or uncomfortable.\n"
        "5. Speak in the first person as yourself."
    )
    
    formatted_messages = []
    formatted_messages.append({"role": "system", "content": system_instruction})
    
    if not history:
        # First turn
        formatted_messages.append({
            "role": "user", 
            "content": f"The debate topic is: \"{topic}\". Please present your opening statement."
        })
    else:
        # Build chronological transcript
        transcript = f"Here is the debate transcript so far on the topic \"{topic}\":\n\n"
        for turn in history:
            speaker_name = turn.get("name", "Unknown AI")
            content = turn.get("content", "")
            transcript += f"--- Speaker: {speaker_name} ---\n{content}\n\n"
            
        transcript += f"It is now your turn, {current_model_name}. Present your counter-argument or supporting points."
        formatted_messages.append({"role": "user", "content": transcript})
        
    return formatted_messages

async def clean_stream_filter(generator):
    """Discard all output before actual STANCE marker to strip thinking/preambles"""
    buffer = ""
    yielded_stance = False
    
    async for chunk in generator:
        if yielded_stance:
            yield chunk
            continue
            
        buffer += chunk
        lower_buf = buffer.lower()
        
        # Look for explicit stance markers (with colons or stars) to avoid false matches in thinking texts
        idx = -1
        for marker in ["**stance**:", "stance:", "**stance**", "stance :"]:
            if marker in lower_buf:
                idx = lower_buf.find(marker)
                break
                
        if idx != -1:
            # Check if there is a "**" before stance
            stars_idx = buffer.rfind("**", 0, idx)
            start_pos = stars_idx if (stars_idx != -1 and idx - stars_idx <= 3) else idx
            yielded_stance = True
            yield buffer[start_pos:]
            buffer = ""
        else:
            if len(buffer) > 1200:
                # Fallback if stance keyword is not found within 1200 characters (thinking block is huge)
                yielded_stance = True
                yield buffer
                buffer = ""

@app.post("/debate/next")
async def debate_next_endpoint(request: Request):
    """Generates the next turn in the AI Council debate"""
    try:
        data = await request.json()
        topic = data.get("topic", "")
        history = data.get("history", [])
        model_id = data.get("model_id", "")
        
        # Calculate turns left (limit to 20 messages total)
        turns_left = max(1, 20 - len(history))
        
        if model_id not in MODEL_GENERATORS:
            return "Invalid Model ID"
            
        generator_func, model_name = MODEL_GENERATORS[model_id]
        messages = build_debate_messages(topic, history, model_id, model_name, turns_left)
        
        return StreamingResponse(
            clean_stream_filter(generator_func(messages)), 
            media_type="text/plain",
            headers={
                "X-Speaker-Id": model_id,
                "X-Speaker-Name": model_name
            }
        )
    except Exception as e:
        return f"Debate Turn Error: {str(e)}"

@app.post("/debate/verdict")
async def debate_verdict_endpoint(request: Request):
    """Summarizes and synthesizes the entire debate into a final verdict"""
    try:
        data = await request.json()
        topic = data.get("topic", "")
        history = data.get("history", [])
        
        # Build the final debate transcript
        transcript = f"Debate Topic: \"{topic}\"\n\n"
        for turn in history:
            speaker_name = turn.get("name", "Unknown AI")
            content = turn.get("content", "")
            transcript += f"[{speaker_name}]: {content}\n\n"
            
        system_instruction = (
            "You are the Scribe of the AI Council. Your job is to listen to the debate transcript "
            "and write a structured, beautifully formatted final verdict.\n"
            "Format your output in Markdown with the following sections:\n"
            "1. **Core Conflict**: A summary of the central debate tension.\n"
            "2. **Key Arguments Pro**: Highlight the main supporting points.\n"
            "3. **Key Arguments Con**: Highlight the main counter-points.\n"
            "4. **Areas of Consensus**: What did the models agree on?\n"
            "5. **Unresolved Disputes**: Where was the disagreement most intense?\n"
            "6. **Final Resolution**: Synthesize the debate into a final compromise or balanced verdict."
        )
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Here is the transcript of the debate:\n\n{transcript}\n\nScribe, please compile your final verdict."}
        ]
        
        return StreamingResponse(generate_gemini(messages), media_type="text/plain")
        
    except Exception as e:
        return f"Verdict Error: {str(e)}"