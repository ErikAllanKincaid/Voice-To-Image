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
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from scipy.io import wavfile

# Config
SAMPLE_RATE = 16000
DEFAULT_CHROMECAST = "Living Room TV"  # Set to None to require explicit device

# Model presets
PRESETS = {
    "lite": {
        "whisper": "tiny",
        "ollama": "llama3.2:1b",
        "sd": "stabilityai/sd-turbo",
        "sd_steps": 4,
    },
    "standard": {
        "whisper": "base",
        "ollama": "llama3.2",
        "sd": "stabilityai/sd-turbo",
        "sd_steps": 4,
    },
    "high": {
        "whisper": "base",
        "ollama": "llama3.2",
        "sd": "stabilityai/sdxl-turbo",
        "sd_steps": 4,
    },
}
DEFAULT_PRESET = "standard"

app = FastAPI(title="Voice-to-Image API")

# Global models (loaded on first use, keyed by model name)
_whisper_models = {}
_sd_pipes = {}


def get_whisper(model_name: str = "base"):
    global _whisper_models
    if model_name not in _whisper_models:
        print(f"Loading Whisper model: {model_name}...")
        _whisper_models[model_name] = WhisperModel(
            model_name,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    return _whisper_models[model_name]


def get_sd_pipe(model_id: str = "stabilityai/sd-turbo"):
    global _sd_pipes
    if model_id not in _sd_pipes:
        print(f"Loading Stable Diffusion model: {model_id}...")
        # Use SDXL pipeline for SDXL models
        pipeline_class = StableDiffusionXLPipeline if "xl" in model_id.lower() else StableDiffusionPipeline
        _sd_pipes[model_id] = pipeline_class.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        if torch.cuda.is_available():
            _sd_pipes[model_id] = _sd_pipes[model_id].to("cuda")
    return _sd_pipes[model_id]


def unload_whisper():
    """Unload Whisper models to free VRAM before loading diffusion."""
    global _whisper_models
    import gc
    _whisper_models.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def transcribe_audio(audio: np.ndarray, whisper_model: str = "base") -> str:
    """Transcribe audio using faster-whisper."""
    # Save to temp WAV for faster-whisper
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wavfile.write(f.name, SAMPLE_RATE, (audio * 32767).astype(np.int16))
        temp_path = f.name

    model = get_whisper(whisper_model)
    segments, _ = model.transcribe(temp_path)
    text = " ".join(seg.text for seg in segments).strip()

    Path(temp_path).unlink()
    return text


def refine_prompt(text: str, ollama_model: str = "llama3.2") -> str:
    """Use Ollama to convert speech to an image generation prompt."""
    system = """Convert to a Stable Diffusion prompt. STRICT LIMIT: 60 words max.

Include: subject, style (cinematic/digital art/photorealistic), lighting, mood.
Add: "highly detailed, 8k" at end.

Be concise. No filler words. Output ONLY the prompt, nothing else."""

    response = ollama.chat(
        model=ollama_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        keep_alive=0,  # Unload model immediately to free VRAM for diffusion
    )
    return response["message"]["content"].strip()


def generate_image(prompt: str, width: int = 768, height: int = 432,
                   sd_model: str = "stabilityai/sd-turbo", sd_steps: int = 4) -> Image.Image:
    """Generate image using Stable Diffusion."""
    # Truncate prompt to ~70 words to stay under CLIP's 77 token limit
    words = prompt.split()
    if len(words) > 70:
        prompt = " ".join(words[:70])

    pipe = get_sd_pipe(sd_model)
    # sd-turbo uses guidance_scale=0, others use 7.5
    guidance = 0.0 if "turbo" in sd_model else 7.5
    image = pipe(prompt, num_inference_steps=sd_steps, guidance_scale=guidance, width=width, height=height).images[0]
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
    preset: str = Form("standard"),
    size: str = Form("768x432"),
):
    """Full pipeline: audio → transcribe → refine → generate → (optional) cast."""
    # Parse size
    try:
        width, height = map(int, size.split("x"))
    except ValueError:
        width, height = 768, 432

    # Get preset config
    config = PRESETS.get(preset, PRESETS[DEFAULT_PRESET])
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
    text = transcribe_audio(data, whisper_model=config["whisper"])
    if not text:
        raise HTTPException(400, "No speech detected")

    # Free Whisper VRAM before loading diffusion model
    unload_whisper()

    prompt = refine_prompt(text, ollama_model=config["ollama"])
    image = generate_image(prompt, width=width, height=height,
                           sd_model=config["sd"], sd_steps=config["sd_steps"])

    # Cast if requested
    if cast:
        cast_device = device if device else DEFAULT_CHROMECAST
        cast_to_chromecast(image, cast_device)

    # Return image + metadata
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    # Free VRAM after generation (trade-off: slower next run, but frees 9GB)
    unload_models()

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
    import gc
    global _whisper_models, _sd_pipes
    _whisper_models.clear()
    _sd_pipes.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"status": "models unloaded"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
