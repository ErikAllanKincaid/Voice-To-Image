#!/usr/bin/env python3
"""Voice-to-Image API Server: Receives audio, returns generated image."""

import io
import tempfile
import time
from pathlib import Path

import numpy as np
import ollama
import torch
from faster_whisper import WhisperModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from scipy.io import wavfile

# Config
WHISPER_MODEL = "base"
OLLAMA_MODEL = "llama3.2"
SD_MODEL = "stabilityai/sd-turbo"
SAMPLE_RATE = 16000
DEFAULT_CHROMECAST = "Living Room TV"  # Set to None to require explicit device

app = FastAPI(title="Voice-to-Image API")

# Global models (loaded on first use)
_whisper_model = None
_sd_pipe = None


def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper model...")
        _whisper_model = WhisperModel(
            WHISPER_MODEL,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    return _whisper_model


def get_sd_pipe():
    global _sd_pipe
    if _sd_pipe is None:
        print("Loading Stable Diffusion model...")
        _sd_pipe = StableDiffusionPipeline.from_pretrained(
            SD_MODEL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        if torch.cuda.is_available():
            _sd_pipe = _sd_pipe.to("cuda")
    return _sd_pipe


def transcribe_audio(audio: np.ndarray) -> str:
    """Transcribe audio using faster-whisper."""
    # Save to temp WAV for faster-whisper
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wavfile.write(f.name, SAMPLE_RATE, (audio * 32767).astype(np.int16))
        temp_path = f.name

    model = get_whisper()
    segments, _ = model.transcribe(temp_path)
    text = " ".join(seg.text for seg in segments).strip()

    Path(temp_path).unlink()
    return text


def refine_prompt(text: str) -> str:
    """Use Ollama to convert speech to an image generation prompt."""
    system = """You are a prompt engineer for image generation.
Convert the user's spoken description into a concise, vivid image prompt.
Focus on visual details: subject, style, lighting, colors, composition.
Output ONLY the prompt, nothing else. Keep it under 77 tokens."""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
    )
    return response["message"]["content"].strip()


def generate_image(prompt: str, width: int = 768, height: int = 432) -> Image.Image:
    """Generate image using Stable Diffusion."""
    pipe = get_sd_pipe()
    image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0, width=width, height=height).images[0]
    return image


def cast_to_chromecast(image: Image.Image, device: str | None = None):
    """Cast image to Chromecast using catt (non-blocking)."""
    import subprocess
    import threading
    import time

    # Save to temp file
    temp_path = Path(tempfile.mktemp(suffix=".png"))
    image.save(temp_path)

    def run_catt():
        cmd = ["catt"]
        if device:
            cmd.extend(["-d", device])
        cmd.extend(["cast", str(temp_path)])
        # Run catt - it will serve the file until interrupted
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Wait for cast to initiate, then kill after 10s (image should be loaded by then)
        time.sleep(10)
        proc.terminate()
        # Clean up temp file
        temp_path.unlink(missing_ok=True)

    thread = threading.Thread(target=run_catt, daemon=True)
    thread.start()


@app.get("/health")
def health():
    return {"status": "ok", "gpu": torch.cuda.is_available()}


@app.post("/transcribe")
async def api_transcribe(audio: UploadFile = File(...)):
    """Transcribe audio to text."""
    content = await audio.read()

    # Parse WAV
    audio_io = io.BytesIO(content)
    try:
        sr, data = wavfile.read(audio_io)
    except Exception as e:
        raise HTTPException(400, f"Invalid WAV file: {e}")

    # Convert to float32 mono
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Resample if needed
    if sr != SAMPLE_RATE:
        from scipy import signal
        data = signal.resample(data, int(len(data) * SAMPLE_RATE / sr))

    text = transcribe_audio(data)
    return {"text": text}


@app.post("/refine")
async def api_refine(text: str = Form(...)):
    """Refine text into an image prompt."""
    prompt = refine_prompt(text)
    return {"prompt": prompt}


@app.post("/generate")
async def api_generate(prompt: str = Form(...)):
    """Generate image from prompt."""
    image = generate_image(prompt)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/pipeline")
async def api_pipeline(
    audio: UploadFile = File(...),
    cast: bool = Form(False),
    device: str = Form(None),
    size: str = Form("768x432"),
):
    """Full pipeline: audio → transcribe → refine → generate → (optional) cast."""
    # Parse size
    try:
        width, height = map(int, size.split("x"))
    except ValueError:
        width, height = 768, 432
    content = await audio.read()

    # Parse WAV
    audio_io = io.BytesIO(content)
    try:
        sr, data = wavfile.read(audio_io)
    except Exception as e:
        raise HTTPException(400, f"Invalid WAV file: {e}")

    # Convert to float32 mono
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Resample if needed
    if sr != SAMPLE_RATE:
        from scipy import signal
        data = signal.resample(data, int(len(data) * SAMPLE_RATE / sr))

    # Pipeline
    text = transcribe_audio(data)
    if not text:
        raise HTTPException(400, "No speech detected")

    prompt = refine_prompt(text)
    image = generate_image(prompt, width=width, height=height)

    # Cast if requested
    if cast:
        cast_device = device if device else DEFAULT_CHROMECAST
        cast_to_chromecast(image, cast_device)

    # Return image + metadata
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "X-Transcription": text,
            "X-Prompt": prompt,
        },
    )


@app.post("/cast")
async def api_cast(image: UploadFile = File(...), device: str = Form(None)):
    """Cast an image to Chromecast."""
    content = await image.read()
    img = Image.open(io.BytesIO(content))
    cast_to_chromecast(img, device if device else None)
    return {"status": "cast complete"}


@app.post("/unload")
def unload_models():
    """Unload models to free VRAM."""
    global _whisper_model, _sd_pipe
    _whisper_model = None
    _sd_pipe = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"status": "models unloaded"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
