

import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from retriever import Retriever
from dotenv import load_dotenv
import ollama
from speech_module import router as speech_router
import whisper

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:1b")

app = FastAPI()


# Add the router
app.include_router(speech_router)

# --- Enable CORS so frontend can call backend ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to your frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load Whisper model once (can be 'tiny', 'base', 'small', 'medium', 'large')
model = whisper.load_model("small")




retriever = Retriever()

class Query(BaseModel):
    question: str

def build_prompt(question, contexts):
    system = "You are a kind, factual assistant specialising in adolescent menstrual health. Use the provided context to answer briefly and clearly. If unsure, encourage seeking a healthcare professional."
    ctx_text = "\n\n".join([f"Source: {c['title']}\n{c['text']}" for c in contexts])
    prompt = f"{system}\n\nContext:\n{ctx_text}\n\nUser: {question}\nAssistant:"
    return prompt

@app.get("/")
async def root():
    return {"message": "Welcome to the Menstrual Health Chatbot API!"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(q: Query):
    contexts = retriever.retrieve(q.question, k=4)
    prompt = build_prompt(q.question, contexts)

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.message.content
        sources = [c["title"] for c in contexts]
        return {"answer": answer, "sources": sources}

    except Exception as e:
        return {"error": str(e)}
    

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(await file.read())

    # Transcribe using Whisper
    result = model.transcribe("temp_audio.wav")

    # Remove temp file
    os.remove("temp_audio.wav")

    return {"text": result["text"]}
"""
# USING GEMINI API
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from retriever import Retriever
from speech_module import router as speech_router
import whisper
import google.generativeai as genai

# --- Load environment ---
#load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

load_dotenv()

# --- Debug check ---
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    print("✅ GOOGLE_API_KEY loaded successfully (length:", len(api_key), ")")
else:
    print("❌ GOOGLE_API_KEY not found! Check your Render environment settings.")

print("GEMINI_API_KEY:", os.getenv("GOOGLE_API_KEY"))

# --- Configure Gemini ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ✅ Create Gemini model instance
gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")

# Initialize FastAPI app
app = FastAPI()
print("🚀 FastAPI app is starting up...")

# Add speech routes
app.include_router(speech_router)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change later to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy load Whisper
whisper_model = None
retriever = Retriever()


class Query(BaseModel):
    question: str


def build_prompt(question, contexts):
    system = (
       "You are a warm, supportive, and knowledgeable assistant who helps young people "
        "understand menstrual health in a clear and comforting way. "
        "Use a friendly and reassuring tone — like someone explaining gently to a friend. "
        "Keep answers concise, relatable, and positive. "
        "If something is unclear or potentially serious, kindly suggest speaking to a healthcare professional. "
        "Avoid being overly formal or robotic."
    )
    ctx_text = "\n\n".join([f"Source: {c['title']}\n{c['text']}" for c in contexts])
    prompt = f"{system}\n\nContext:\n{ctx_text}\n\nUser: {question}\nAssistant:"
    return prompt


@app.get("/")
async def root():
    return {"message": "Welcome to the Menstrual Health Chatbot API!"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(q: Query):
    contexts = retriever.retrieve(q.question, k=4)
    prompt = build_prompt(q.question, contexts)

    try:
        # ✅ Use Gemini model here (not Whisper)
        response = gemini_model.generate_content(prompt)
        answer = response.text if hasattr(response, "text") else str(response)
        sources = [c["title"] for c in contexts]
        return {"answer": answer, "sources": sources}
    except Exception as e:
        print("Error generating response:", str(e))
        return {"error": str(e)}


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model("small")

    with open("temp_audio.wav", "wb") as f:
        f.write(await file.read())

    result = whisper_model.transcribe("temp_audio.wav")
    os.remove("temp_audio.wav")

    return {"text": result["text"]}


if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting server on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port)

"""