"""
JARVIS — Voice AI Assistant Backend
=====================================
Groq API (primary, ultra-fast) + Claude Haiku (fallback/complex tasks)
FastAPI + WebSocket + ElevenLabs TTS + AppleScript modules

Setup:
  pip install fastapi uvicorn websockets groq anthropic elevenlabs python-dotenv

.env file:
  GROQ_API_KEY=gsk_...
  ANTHROPIC_API_KEY=sk-ant-...
  ELEVENLABS_API_KEY=...
  ELEVENLABS_VOICE_ID=...   # British voice ID

Run:
  python -m uvicorn jarvis_backend_groq:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
import base64
import asyncio
import sys
import subprocess
import webbrowser
import sqlite3
import datetime
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import AsyncGroq
import anthropic

from fastapi.staticfiles import StaticFiles

load_dotenv()

app = FastAPI(title="JARVIS Voice AI")

# Serve static files (like your background image)
app.mount("/static", StaticFiles(directory="."), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Clients ───────────────────────────────────────────────────────────────────

groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
claude_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

GROQ_MODEL = "llama-3.3-70b-versatile"   # fastest + smartest
GROQ_FAST  = "llama-3.1-8b-instant"            # ultra-fast for simple queries
CLAUDE_MODEL = "claude-3-5-haiku-latest"

ELEVENLABS_KEY     = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE   = os.getenv("ELEVENLABS_VOICE_ID", "")

SYSTEM_PROMPT = """You are JARVIS, a sharp British voice assistant for your creator, Vaibhav.
You speak concisely — short sentences, no filler words, no markdown.
Your tone is professional, slightly witty, and highly efficient.
Always address the user as 'Vaibhav' or 'Sir'.
You can check the calendar, read emails, create notes, browse the web, and control the system.
If you open a website or app, confirm it with a short, sharp phrase."""

# ─── Memory (SQLite FTS5) ───────────────────────────────────────────────────────

DB_PATH = "jarvis_memory.db"

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memory USING fts5(
            role, content, ts UNINDEXED
        )
    """)
    con.commit()
    con.close()

def save_memory(role: str, content: str):
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT INTO memory VALUES (?, ?, ?)",
                (role, content, datetime.datetime.utcnow().isoformat()))
    con.commit()
    con.close()

def search_memory(query: str, limit: int = 6) -> list[dict]:
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT role, content, ts FROM memory WHERE memory MATCH ? ORDER BY rank LIMIT ?",
        (query, limit)
    ).fetchall()
    con.close()
    return [{"role": r[0], "content": r[1], "ts": r[2]} for r in rows]

# ─── AppleScript helpers ────────────────────────────────────────────────────────

def run_applescript(script: str) -> str:
    """Helper to run AppleScript (macOS only)."""
    if sys.platform != "darwin":
        return "This feature (AppleScript) is only available on macOS."
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() or result.stderr.strip()
    except Exception as e:
        return f"AppleScript error: {e}"

def get_calendar_events(days: int = 3) -> str:
    script = f"""
    set output to ""
    set d to current date
    set endDate to d + ({days} * days)
    tell application "Calendar"
        repeat with cal in calendars
            set evts to (every event of cal whose start date >= d and start date <= endDate)
            repeat with e in evts
                set output to output & summary of e & " @ " & (start date of e as string) & "\\n"
            end repeat
        end repeat
    end tell
    return output
    """
    return run_applescript(script) or "No upcoming events."

def get_mail_summary(count: int = 5) -> str:
    script = f"""
    set output to ""
    tell application "Mail"
        set msgs to (messages of inbox whose read status is false)
        set cnt to count of msgs
        if cnt > {count} then set cnt to {count}
        repeat with i from 1 to cnt
            set m to item i of msgs
            set output to output & subject of m & " — from " & sender of m & "\\n"
        end repeat
    end tell
    return output
    """
    return run_applescript(script) or "No unread emails."

def create_note(title: str, body: str) -> str:
    script = f"""
    tell application "Notes"
        make new note at folder "Notes" with properties {{name:"{title}", body:"{body}"}}
    end tell
    return "Note created."
    """
    return run_applescript(script)

def open_url(url: str) -> str:
    if os.name == "nt": # Windows
        os.startfile(url)
        return f"Opened {url}"
    else: # macOS/Linux
        script = f'open location "{url}"'
        return run_applescript(script) or f"Opened {url}"

# ─── AI Router ─────────────────────────────────────────────────────────────────

def classify_query(text: str) -> str:
    """Route to the right model based on query type."""
    text_lower = text.lower()

    # Complex coding / long research → Claude
    if any(k in text_lower for k in ["build an app", "write code", "create a script",
                                      "research", "explain in detail", "compare"]):
        return "claude"

    # Fast conversational → Groq fast model
    if len(text.split()) < 12:
        return "groq_fast"

    # Default → Groq flagship
    return "groq"

async def ask_groq(messages: list, fast: bool = False) -> str:
    model = GROQ_FAST if fast else GROQ_MODEL
    resp = await groq_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=512,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

async def ask_claude(messages: list) -> str:
    # Convert to claude format (strip system from messages list)
    user_msgs = [m for m in messages if m["role"] != "system"]
    resp = await claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=user_msgs,
    )
    return resp.content[0].text.strip()

async def get_ai_response(user_text: str, history: list) -> str:
    route = classify_query(user_text)

    # Build message list with system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    # Check for action commands first
    text_lower = user_text.lower()

    if any(k in text_lower for k in ["calendar", "schedule", "events", "appointments"]):
        data = get_calendar_events()
        messages.append({"role": "user", "content": f"Here are my calendar events:\n{data}\n\nUser said: {user_text}"})
    elif any(k in text_lower for k in ["email", "mail", "inbox", "unread"]):
        data = get_mail_summary()
        messages.append({"role": "user", "content": f"Here are unread emails:\n{data}\n\nUser said: {user_text}"})
    elif "play" in text_lower and ("song" in text_lower or "music" in text_lower or "on youtube" in text_lower):
        query = text_lower.replace("play", "").replace("on youtube", "").replace("song", "").strip()
        url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
        webbrowser.open(url)
        return f"Searching YouTube and playing the top result for {query}."
    elif "search youtube for" in text_lower:
        query = text_lower.replace("search youtube for", "").strip()
        url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
        webbrowser.open(url)
        return f"I've searched YouTube for {query}."
    elif "open whatsapp" in text_lower:
        webbrowser.open("https://web.whatsapp.com")
        return "Opening WhatsApp Web."
    elif "open gmail" in text_lower:
        webbrowser.open("https://mail.google.com")
        return "Accessing your Gmail, sir."
    elif "open chrome" in text_lower or "open google" in text_lower:
        webbrowser.open("https://www.google.com")
        return "Opening Chrome."
    elif "open maps" in text_lower or "google maps" in text_lower:
        webbrowser.open("https://maps.google.com")
        return "Navigating via Google Maps."
    elif "open notepad" in text_lower:
        if os.name == "nt":
            subprocess.Popen(["notepad.exe"])
            return "Opening Notepad for you, sir."
        else:
            return "Notepad is not available on this system, sir."
    elif any(k in text_lower for k in ["search for", "look up", "google search"]):
        query = text_lower.replace("search for", "").replace("look up", "").replace("google search", "").strip()
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        webbrowser.open(url)
        return f"I've searched Google for {query}. The results are on your screen."
    elif text_lower.startswith("note:") or "create a note" in text_lower:
        parts = user_text.split(":", 1)
        title = "JARVIS Note"
        body = parts[1].strip() if len(parts) > 1 else user_text
        result = create_note(title, body)
        messages.append({"role": "user", "content": f"{result}. User said: {user_text}"})
    elif any(k in text_lower for k in ["open ", "browse ", "go to ", "search google"]):
        if "google" in text_lower:
            query = user_text.lower().replace("search google for", "").replace("google", "").strip()
            url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        else:
            url = "https://www.google.com"
        open_url(url)
        messages.append({"role": "user", "content": f"Opened browser for: {user_text}"})
    else:
        messages.append({"role": "user", "content": user_text})

    try:
        if route == "claude":
            return await ask_claude(messages)
        elif route == "groq_fast":
            return await ask_groq(messages, fast=True)
        else:
            return await ask_groq(messages, fast=False)
    except Exception as e:
        # Fallback chain: Groq → Claude → error
        try:
            if route != "claude":
                return await ask_claude(messages)
        except Exception:
            pass
        return f"I encountered an error: {e}"

# ─── ElevenLabs TTS ────────────────────────────────────────────────────────────

async def text_to_speech_elevenlabs(text: str) -> bytes | None:
    if not ELEVENLABS_KEY or not ELEVENLABS_VOICE:
        return None
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE}"
    headers = {"xi-api-key": ELEVENLABS_KEY, "Content-Type": "application/json"}
    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, headers=headers, timeout=15)
        if resp.status_code == 200:
            return resp.content
    return None

def text_to_speech_fallback(text: str):
    """Fallback TTS: macOS 'say' or Windows PowerShell speech."""
    if os.name == "nt":
        # Windows PowerShell speech
        clean_text = text.replace('"', "'")
        cmd = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{clean_text}")'
        subprocess.Popen(["powershell", "-Command", cmd])
    else:
        # macOS fallback
        subprocess.Popen(["say", "-v", "Daniel", text])

# ─── WebSocket Handler ──────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    history: list[dict] = []

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            user_text: str = data.get("text", "").strip()

            if not user_text:
                continue

            # Send thinking state
            await ws.send_json({"type": "state", "state": "thinking"})

            # Retrieve relevant memories
            memories = search_memory(user_text)
            mem_context = "\n".join(f"[{m['ts']}] {m['role']}: {m['content']}" for m in memories)
            if mem_context:
                history_with_mem = [{"role": "system", "content": f"Relevant past context:\n{mem_context}"}] + history
            else:
                history_with_mem = history

            # Get AI response
            reply = await get_ai_response(user_text, history_with_mem)

            # Update history (keep last 10 turns)
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": reply})
            history = history[-20:]

            # Save to memory
            save_memory("user", user_text)
            save_memory("assistant", reply)

            # Send text response
            await ws.send_json({"type": "text", "text": reply})

            # TTS
            await ws.send_json({"type": "state", "state": "speaking"})
            audio = await text_to_speech_elevenlabs(reply)
            if audio:
                audio_b64 = base64.b64encode(audio).decode()
                await ws.send_json({"type": "audio", "audio": audio_b64, "format": "mp3"})
            else:
                # Fallback TTS
                text_to_speech_fallback(reply)

            await ws.send_json({"type": "state", "state": "idle"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await ws.send_json({"type": "error", "error": str(e)})

# ─── REST endpoints ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    with open("jarvis_frontend.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "groq": bool(os.getenv("GROQ_API_KEY")),
        "claude": bool(os.getenv("ANTHROPIC_API_KEY")),
        "elevenlabs": bool(ELEVENLABS_KEY),
    }

@app.get("/memory/search")
async def memory_search(q: str, limit: int = 10):
    return search_memory(q, limit)

@app.delete("/memory/clear")
async def memory_clear():
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM memory")
    con.commit()
    con.close()
    return {"cleared": True}

# ─── Startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    init_db()
    print("✅ JARVIS backend started")
    print(f"   Groq model   : {GROQ_MODEL}")
    print(f"   Claude model : {CLAUDE_MODEL}")
    print(f"   ElevenLabs   : {'✓' if ELEVENLABS_KEY else '✗ (macOS say fallback)'}")
