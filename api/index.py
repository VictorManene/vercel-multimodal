#import libraries
import os
import time
import uuid
import logging
import requests
import subprocess
from dotenv import load_dotenv
import cv2
import pytesseract
import pyttsx3 
import pygame    
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from deep_translator import GoogleTranslator
from langdetect import detect
from langchain.agents import AgentExecutor, Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain import hub
import threading,ast
import pytesseract
from pytesseract import TesseractNotFoundError
import glob
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class AnalyzeRequest(BaseModel):
    url: str
    question: str

@app.post("/api/multimodal")
async def multimodal_endpoint(body: AnalyzeRequest):
    transcript, ocr, answer, summary, _ = process_video(
        body.url, body.question, play_audio=False
    )
    return {
        "transcript": transcript,
        "ocr": ocr,
        "answer": answer,
        "summary": summary,
    }

class AgentRequest(BaseModel):
    url: str
    prompt: str

@app.post("/api/agent")
async def agent_endpoint(body: AgentRequest):
    # ensure your agent is already initialized at the module level
    result = create_agent_system().invoke({
        "input": f"Watch this video and tell me about it: {body.url}\n{body.prompt}"
    })
    return {"output": result}


# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Loading Environment Variables
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    raise EnvironmentError("Set YOUTUBE_API_KEY in your .env")
COOKIE_FILE = os.getenv("YT_COOKIES_PATH", "www.youtube.com_cookies.txt").strip()
logging.info(f"YT_COOKIES_PATH → {COOKIE_FILE!r}")
logging.info(f"Exists? {os.path.exists(COOKIE_FILE)}")


#API & Model Config
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Directories
OUTPUT_DIR = "files/audio"
VIDEO_DIR  = "files/video"
TEXT_DIR   = "files/text"
DB_DIR     = "files/chroma_db"
for d in (OUTPUT_DIR, VIDEO_DIR, TEXT_DIR, DB_DIR):
    os.makedirs(d, exist_ok=True)

# Lazy global variables
_transcriber = None
_embedder   = None
_splitter   = None
_http_sess  = None
_vectorstore= None

# Initialize vector
def load_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embedding_fn = SentenceTransformerEmbeddings(model_name="paraphrase-MiniLM-L3-v2")
        _vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embedding_fn)
    return _vectorstore
# Load or return Whisper transcription model
def get_transcriber():
    global _transcriber
    if _transcriber is None:
        _transcriber = WhisperModel(model_size_or_path="small", device="cpu", compute_type="int8")
    return _transcriber
# Load or return sentence embedding model
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    return _embedder
# Load or return text splitter
def get_splitter():
    global _splitter
    if _splitter is None:
        _splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return _splitter
# Load or return HTTP session with auth headers
def get_http_session():
    global _http_sess
    if _http_sess is None:
        _http_sess = requests.Session()
        _http_sess.headers.update({
            "Authorization": f"Bearer {YOUTUBE_API_KEY}",
            "Content-Type": "application/json"
        })
    return _http_sess

# Text-to-Speech (WAV) 
def text_to_speech(text: str, output_path: str):
    # Save as WAV for pygame compatibility
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1)
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    logger.info(f"TTS saved to {output_path}")

def fetch_public_transcript(video_id: str) -> str:
    """
    Fetches the publicly available transcript for a YouTube video.
    Returns a plain text string (all segments joined), or raises if none exists.
    """
    try:
        raw = YouTubeTranscriptApi.get_transcript(video_id)
        # raw is a list of dicts: [{ "text": ..., "start": ..., "duration": ... }, …]
        return " ".join(seg["text"].strip() for seg in raw)
    except NoTranscriptFound:
        raise RuntimeError(f"No public transcript available for video {video_id}")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch transcript: {e}")

# Download Audio from Youtube as WAV

def download_audio_only(url: str):
    vid_id = uuid.uuid4().hex
    base = os.path.join(OUTPUT_DIR, vid_id)

    # Base yt-dlp options
    opts = {
        'format': 'bestaudio/best',
        'http_headers': {'User-Agent': 'Mozilla/5.0'},
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'postprocessor_args': ['-ar', '16000', '-ac', '1'],
        'outtmpl': base + '.%(ext)s',
        'quiet': True,
    }

    # Attach cookies if present
    if COOKIE_FILE and os.path.exists(COOKIE_FILE):
        logging.info(f"Using cookiefile at {COOKIE_FILE}")
        opts['cookiefile'] = COOKIE_FILE

    try:
        # Try to download & convert to WAV
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)

        wav_path = base + '.wav'
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Audio missing: {wav_path}")

        # Success: return local WAV path
        return wav_path, vid_id, None

    except DownloadError as first_exc:
        logging.warning(f"Full download failed ({first_exc}), trying metadata‐only...")

        # 2) Try metadata‐only to get a stream URL
        meta_opts = {
            'format': 'bestaudio/best',
            'quiet': True,
        }
        if COOKIE_FILE and os.path.exists(COOKIE_FILE):
            logging.info(f"Using cookiefile for metadata: {COOKIE_FILE}")
            meta_opts['cookiefile'] = COOKIE_FILE

        try:
            with YoutubeDL(meta_opts) as ydl2:
                info = ydl2.extract_info(url, download=False)
            stream_url = info.get('url')
            if stream_url:
                return None, vid_id, stream_url
            else:
                logging.warning("Metadata‐only returned no URL field")
        except DownloadError as meta_exc:
            logging.warning(f"Metadata‐only also failed ({meta_exc})")

        # 3) Give up on yt-dlp entirely
        logging.error("yt-dlp cannot fetch or download. Falling back to transcript‐only.")
        return None, vid_id, None


def watch_video_tool(url: str):
    """
    Download both audio (for transcription) and video (for OCR) from YouTube.
    Returns WAV path and video ID.
    """
    # download_audio_only now returns (audio_path, vid, stream_url)
    audio_path, vid, _ = download_audio_only(url)

    video_out = os.path.join(VIDEO_DIR, f"{vid}.mp4")
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': video_out,
        'merge_output_format': 'mp4',
        'quiet': True,
    }
    if COOKIE_FILE and os.path.exists(COOKIE_FILE):
        ydl_opts['cookiefile'] = COOKIE_FILE

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        if not os.path.exists(video_out):
            raise FileNotFoundError(f"Video missing: {video_out}")
    except DownloadError as e:
        logging.warning(f"Video download failed ({e}), skipping OCR/video analysis.")
        # We’ll just return audio_path and vid; downstream OCR will see no video file
        return audio_path, vid
    
    return audio_path, vid


# OCR frames of video at given interval
def ocr_video_frames(video_path: str, interval_s: float = 10.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    interval = max(int(fps * interval_s), 1)
    results = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            if text.strip():
                results.append((idx / fps, text.strip()))
        idx += 1
    cap.release()
    return results

# Transcribe 30s audio chunks via whisper
def _transcribe_chunk(args):
    path, model = args
    segs, _ = model.transcribe(path, beam_size=1, language=None)
    return " ".join(s.text.strip() for s in segs)
#main transciption function
def transcribe_audio(audio_path: str):
    model = get_transcriber()
    info = subprocess.check_output([
        "ffprobe", "-i", audio_path,
        "-show_entries", "format=duration",
        "-v", "quiet", "-of", "csv=p=0"
    ]).decode().strip()
    total = float(info)
    chunks = []
    for i in range(int(np.ceil(total / 30))):
        start = i * 30
        out = f"{audio_path}_{i}.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            "-ss", str(start), "-t", "30",
            "-acodec", "pcm_s16le", "-ac", "1", out
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        chunks.append(out)
    with ThreadPoolExecutor(max_workers=4) as exe:
        texts = exe.map(_transcribe_chunk, [(c, model) for c in chunks])
    for c in chunks:
        os.remove(c)
    return " ".join(texts)

# Embedding & Store transcript + OCR in vectorstore
def embed_and_persist(transcript: str, ocr_data: list, vid: str):
    chunks = get_splitter().split_text(transcript)
    ocr_chunks = [f"[{int(t)}s] {txt}" for t, txt in ocr_data]
    docs = chunks + ocr_chunks
    embeds = get_embedder().encode(docs, batch_size=16, normalize_embeddings=True)
    store = load_vectorstore()
    ids = [f"{vid}_{i}" for i in range(len(docs))]
    metas = []
    for i, doc in enumerate(docs):
        if i < len(chunks):
            metas.append({"video_id": vid, "type": "transcript", "chunk_index": i})
        else:
            metas.append({"video_id": vid, "type": "ocr", "timecode": ocr_data[i-len(chunks)][0]})
    store.add_texts(texts=docs, embeddings=embeds.tolist(), ids=ids, metadatas=metas)

# Retrieval & Query
def retrieve_context(q: str, k: int = 3):
    docs = load_vectorstore().as_retriever(search_type="similarity", search_kwargs={"k": k}).get_relevant_documents(q)
    return "\n\n".join(d.page_content for d in docs)

def query_llm(prompt: str):
    headers = {
        "Authorization": f"Bearer {YOUTUBE_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(GROQ_API_URL, json={
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.7
    }, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

# Cleanup temporary files
def cleanup(paths: list):
    for p in paths:
        if p and os.path.exists(p):
            os.remove(p)

# Full Pipeline
def process_video(url: str, question: str, play_audio: bool = True):
    """
    Process a YouTube video, return transcript, OCR, answer, summary, and optionally play audio.
    :param url: YouTube video URL
    :param question: Query against the video content
    :param play_audio: If True, plays the TTS audio via pygame; otherwise skips playback.
    """
    audio, vid, stream_url  = download_audio_only(url)
    if audio is None and stream_url:
        # Stream into Whisper
        segments, _ = get_transcriber().transcribe(
            stream_url, beam_size=5, language=None
        )
        transcript = " ".join(seg.text.strip() for seg in segments)
        # we skipped OCR/video but still can do summary/answer
        context = retrieve_context(question)
        answer  = query_llm(f"Answer using transcript only:\n{transcript}\nQuestion: {question}")
        summary = query_llm(f"Summarize using transcript only:\n{transcript}")
        return transcript, [], answer, summary, None
    video_path = os.path.join(VIDEO_DIR, f"{vid}.mp4")
    # optional: call watch_video_tool to download video too
    if not os.path.exists(video_path):
        _, _ = watch_video_tool(url)
    # Gracefully handle OCR if video file is available
    if os.path.exists(video_path):
        ocr_data = ocr_video_frames(video_path)
    else:
        logging.warning("No video file available; skipping OCR.")
        ocr_data = []
    transcript = transcribe_audio(audio)
    txt_path = os.path.join(TEXT_DIR, f"transcript_{int(time.time())}_{vid}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    embed_and_persist(transcript, ocr_data, vid)
    context = retrieve_context(question)
    answer = query_llm(f"Answer using context:\n{context}\nQuestion: {question}")
    summary = query_llm(f"Summarize using context:\n{context}")
    if (lang := detect(question)) != 'en':
        answer = GoogleTranslator(source='auto', target=lang).translate(answer)
        summary = GoogleTranslator(source='auto', target=lang).translate(summary)

    # Text-to-Speech 
    tts_path = os.path.join(OUTPUT_DIR, f"{vid}_response.wav")  # use WAV
    text_to_speech(answer, output_path=tts_path)

    # Playback with pygame 
    if play_audio:
        pygame.mixer.init()
        pygame.mixer.music.load(tts_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    cleanup([audio, txt_path])
    return transcript, ocr_data, answer, summary, tts_path

def create_agent_system():


    def summarize_tool(input_data):
        logger.info(f"[Tool] SummarizeContent received: {input_data!r} (type: {type(input_data)})")

        # If the agent didn’t supply args, grab the latest WAV in OUTPUT_DIR
        if input_data is None:
            wavs = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.wav")), key=os.path.getmtime)
            if not wavs:
                raise RuntimeError("No audio files found in OUTPUT_DIR")
            audio_path = wavs[-1]
            vid = os.path.splitext(os.path.basename(audio_path))[0]

        # Otherwise parse explicit input_data as before
        elif isinstance(input_data, str) and input_data.strip().startswith("("):
            try:
                seq = ast.literal_eval(input_data)
                audio_path, vid = seq[0], seq[1]
            except Exception as e:
                raise ValueError(f"Failed to parse tuple string: {e}")

        elif not isinstance(input_data, str) and hasattr(input_data, '__getitem__'):
            try:
                audio_path, vid = input_data[0], input_data[1]
            except Exception as e:
                raise ValueError(f"Failed to unpack sequence: {e}")

        elif isinstance(input_data, str) and "AUDIO_PATH:" in input_data:
            parts = input_data.split("AUDIO_PATH:")[1].split("|VID_ID:")
            audio_path, vid = parts[0], parts[1]

        else:
            raise ValueError(f"Invalid input_data for summarize_tool: {input_data!r}")

        # Transcribe audio
        transcript = transcribe_audio(audio_path)
        # Ensure video downloaded for OCR
        video_path = os.path.join(VIDEO_DIR, f"{vid}.mp4")
        if not os.path.exists(video_path):
            watch_video_tool(f"https://youtu.be/{vid}")

        # Run OCR with tesseract check
        try:
            ocr_data = ocr_video_frames(video_path)
        except TesseractNotFoundError:
            logger.error("Tesseract not found in PATH, skipping OCR.")
            ocr_data = []
        # Persist embeddings
        embed_and_persist(transcript, ocr_data, vid)

        # Build prompt
        ocr_snips = "\n".join(f"[{int(t)}s] {txt}" for t, txt in ocr_data)
        prompt = (
            f"Summarize the following transcript and OCR content:\n"
            f"Transcript:\n{transcript}\n\n"
            f"OCR:\n{ocr_snips}"
        )
        return query_llm(prompt)

    def read_findings_tool(q_vid):
        if isinstance(q_vid, str) and q_vid.strip().startswith("("):
            try:
                seq = ast.literal_eval(q_vid)
                q, vid = seq[0], seq[1]
            except:
                q, vid = str(q_vid), None
        elif not isinstance(q_vid, str) and hasattr(q_vid, '__getitem__'):
            q, vid = q_vid[0], q_vid[1] if len(q_vid) > 1 else None
        else:
            q, vid = str(q_vid), None
        context = retrieve_context(q)
        return query_llm(f"Answer using context:\n{context}\nQuestion: {q}")

    def speak_tool(text: str):
        try:
            lang = detect(text)
        except:
            lang = 'en'
        text_en = GoogleTranslator(source='auto', target='en').translate(text) if lang != 'en' else text
        path = os.path.join(OUTPUT_DIR, f"response_{uuid.uuid4().hex}.wav")
        text_to_speech(text_en, path)
        def _play():
            pygame.mixer.init()
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        threading.Thread(target=_play, daemon=True).start()
        return path

    tools = [
        Tool.from_function(func=watch_video_tool, name="WatchVideo", description="Download audio & video for OCR"),
        Tool.from_function(func=summarize_tool, name="SummarizeContent", description="Summarize transcript and OCR"),
        Tool.from_function(func=read_findings_tool, name="ReadSlides", description="Answer using context"),
        Tool.from_function(func=speak_tool, name="Speak", description="TTS playback")
    ]
    llm = ChatOpenAI(
        model="llama3-8b-8192",
        temperature=0,
        openai_api_key=YOUTUBE_API_KEY,
        openai_api_base="https://api.groq.com/openai/v1"
    )
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


# --- Main Entry with Mode Selection ---
def simulate_login():
    logger.info("User simulated as logged in.")
    return True
is_logged_in=False
def ensure_authenticated():
    if not is_logged_in: raise RuntimeError("Login required")

