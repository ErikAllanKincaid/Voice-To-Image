#!/usr/bin/env python3
"""Voice-to-Image API Server: Receives audio, returns generated image."""

import io
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import ollama
import torch
from faster_whisper import WhisperModel
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, FluxPipeline
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from scipy.io import wavfile

# Config
SAMPLE_RATE = 16000
DEFAULT_CHROMECAST = "Living Room TV"  # Set to None to require explicit device

# Prompt-refinement (LLM) config — env-overridable
REFINE_MODEL = os.environ.get("REFINE_MODEL", "").strip()  # non-empty overrides every preset's "ollama"
REFINE_TEMPERATURE = float(os.environ.get("REFINE_TEMPERATURE", "0.2"))
# qwen emits a <think> block that leaves message.content empty; disable thinking unless forced.
_REFINE_THINK_ENV = os.environ.get("REFINE_THINK", "").strip().lower()
_THINKING_FAMILIES = ("qwen", "deepseek", "magistral", "gpt-oss")

# Per-stage model selection. The Web UI may pick only from these allowlists; an arbitrary
# identifier (sd_model especially) would let a client trigger any Hugging Face download on a
# shared host, so unlisted values are rejected in /pipeline.
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
OLLAMA_MODELS = ["llama3.2:1b", "llama3.2", "qwen3.5:9b"]
# diffusion model -> inference steps; keys are the only accepted sd_model values.
SD_MODELS = {
    "stabilityai/sd-turbo": 4,
    "stabilityai/sdxl-turbo": 4,
    "stabilityai/stable-diffusion-xl-base-1.0": 30,
    "black-forest-labs/FLUX.1-schnell": 4,
}
# Rough peak VRAM (MB), fp16, for the UI pre-warning only. Whisper keys are "whisper:<name>".
VRAM_ESTIMATE_MB = {
    "whisper:tiny": 600, "whisper:base": 900, "whisper:small": 1600,
    "whisper:medium": 3000, "whisper:large-v3": 4700,
    "llama3.2:1b": 1500, "llama3.2": 3200, "qwen3.5:9b": 7000,
    "stabilityai/sd-turbo": 3000, "stabilityai/sdxl-turbo": 7500,
    "stabilityai/stable-diffusion-xl-base-1.0": 9000,
    "black-forest-labs/FLUX.1-schnell": 11000,
}
# Deployment locks: when set, force that model for every request and hide the alternatives.
# REFINE_MODEL (above) is the LLM lock.
WHISPER_LOCK = os.environ.get("WHISPER_MODEL", "").strip()
SD_LOCK = os.environ.get("SD_MODEL", "").strip()

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
        "ollama": "qwen3.5:9b",
        "sd": "stabilityai/sdxl-turbo",
        "sd_steps": 4,
    },
    # 24GB+ VRAM only
    "ultra": {
        "whisper": "medium",
        "ollama": "qwen3.5:9b",
        "sd": "stabilityai/stable-diffusion-xl-base-1.0",
        "sd_steps": 30,
    },
    # FLUX - needs 24GB+ or 2x 12GB GPUs
    "flux": {
        "whisper": "base",
        "ollama": "qwen3.5:9b",
        "sd": "black-forest-labs/FLUX.1-schnell",
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
        print(f"Loading diffusion model: {model_id}...")

        if "flux" in model_id.lower():
            # FLUX models need bfloat16 and CPU offload for reliable VRAM release
            # (device_map="balanced" doesn't release memory properly)
            print("Loading FLUX with CPU offload...")
            _sd_pipes[model_id] = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
            )
            _sd_pipes[model_id].enable_model_cpu_offload()
        else:
            # Standard SD/SDXL models
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


REFINE_SYSTEM = """You rewrite a transcribed spoken description into a single Stable Diffusion image prompt.
- Describe ONLY what the speaker described. Keep every named subject, object, and spatial relationship. Do not invent new subjects or scenes.
- Turn it into concrete visual detail: materials, colours, lighting, composition, mood.
- One line, comma-separated fragments, max 55 words.
- Never refuse. Never add commentary, quotes, or a preamble. Output the prompt text only."""

# Few-shot pairs as real conversation turns. Embedding examples in the system message made
# qwen3.5 emit an immediate end-of-text (empty content); alternating user/assistant turns do not.
REFINE_SHOTS = [
    ("my cat sleeping in a sunny window",
     "tabby cat curled asleep on a wooden windowsill, warm afternoon sunlight through the glass, "
     "soft shadows, potted plants nearby, cosy domestic scene, shallow depth of field, highly detailed"),
    ("a busy market street with lots of stalls and people",
     "crowded open-air market street, rows of colourful fabric-covered stalls, vendors and shoppers "
     "browsing produce, strings of lanterns overhead, warm golden-hour light, bustling atmosphere, "
     "wide street-level composition, highly detailed"),
]

# Leading labels an LLM may prepend despite instructions; matched case-insensitively.
_LABEL_PREFIXES = (
    "here is a stable diffusion image prompt:",
    "here is the image prompt:",
    "here is a prompt:",
    "here is your prompt:",
    "here's a prompt:",
    "sure, here's the prompt:",
    "stable diffusion image prompt:",
    "stable diffusion prompt:",
    "image prompt:",
    "sd prompt:",
    "prompt:",
    "image:",
)
_REFUSAL_RE = re.compile(
    r"\b(i (?:can ?not|can't|won't|will not|am unable|'m unable)|as an ai|i'?m sorry|i am sorry)\b",
    re.IGNORECASE,
)


def _refine_think(model_name: str):
    """Return the `think` value for ollama.chat, or None to omit the kwarg.

    Default: disable thinking for hybrid-reasoning families — for this short rewrite
    task their reasoning block tends to swallow the answer, leaving content empty.
    REFINE_THINK=1/0 overrides.
    """
    if _REFINE_THINK_ENV in ("1", "true", "yes"):
        return True
    if _REFINE_THINK_ENV in ("0", "false", "no"):
        return False
    if model_name.lower().startswith(_THINKING_FAMILIES):
        return False
    return None


def clean_prompt(raw: str, fallback: str) -> str:
    """Normalise LLM output to one prompt line; fall back to the transcript on empty/refusal."""
    text = " ".join((raw or "").split())
    changed = True
    while changed:
        changed = False
        low = text.lower()
        for p in _LABEL_PREFIXES:
            if low.startswith(p):
                text = text[len(p):].strip()
                changed = True
                break
    if len(text) >= 2 and text[0] in "\"'`" and text[-1] == text[0]:
        text = text[1:-1].strip()
    if not text or _REFUSAL_RE.search(text):
        return fallback.strip()
    return text


def refine_prompt(text: str, ollama_model: str = "llama3.2") -> str:
    """Use Ollama to convert transcribed speech into a grounded image-generation prompt."""
    model = REFINE_MODEL or ollama_model
    think = _refine_think(model)
    kwargs = {} if think is None else {"think": think}

    messages = [{"role": "system", "content": REFINE_SYSTEM}]
    for shot_in, shot_out in REFINE_SHOTS:
        messages.append({"role": "user", "content": f'Speech: "{shot_in}"'})
        messages.append({"role": "assistant", "content": shot_out})
    messages.append({"role": "user", "content": f'Speech: "{text}"'})

    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            keep_alive=0,  # Unload model immediately to free VRAM for diffusion
            options={"temperature": REFINE_TEMPERATURE},
            **kwargs,
        )
        raw = response["message"]["content"]
    except Exception as e:
        # Shared GPU: Ollama can OOM or be unavailable. Degrade to the raw transcript
        # rather than 500 the whole pipeline.
        print(f"refine_prompt: ollama.chat failed ({e}); using raw transcript")
        raw = ""
    return clean_prompt(raw, fallback=text)


def generate_image(prompt: str, width: int = 768, height: int = 432,
                   sd_model: str = "stabilityai/sd-turbo", sd_steps: int = 4) -> Image.Image:
    """Generate image using Stable Diffusion."""
    # Truncate prompt to ~70 words to stay under CLIP's 77 token limit
    words = prompt.split()
    if len(words) > 70:
        prompt = " ".join(words[:70])

    pipe = get_sd_pipe(sd_model)
    # turbo and schnell models use guidance_scale=0, others use 7.5
    guidance = 0.0 if ("turbo" in sd_model or "schnell" in sd_model) else 7.5
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


def _resolve_models(preset_cfg, whisper_req, refine_req, sd_req):
    """Pick each stage's model: explicit request field > deployment lock > preset default.

    Non-locked values must be on their allowlist (a free-form sd_model would let a client
    trigger arbitrary Hugging Face downloads on a shared host). Returns
    (whisper_name, refiner_name, sd_name, sd_steps).
    """
    whisper = WHISPER_LOCK or whisper_req or preset_cfg["whisper"]
    if not WHISPER_LOCK and whisper not in WHISPER_MODELS:
        raise HTTPException(400, f"whisper_model not allowed: {whisper!r}")

    refiner = REFINE_MODEL or refine_req or preset_cfg["ollama"]
    if not REFINE_MODEL and refiner not in OLLAMA_MODELS:
        raise HTTPException(400, f"refine_model not allowed: {refiner!r}")

    sd = SD_LOCK or sd_req or preset_cfg["sd"]
    if not SD_LOCK and sd not in SD_MODELS:
        raise HTTPException(400, f"sd_model not allowed: {sd!r}")

    return whisper, refiner, sd, SD_MODELS.get(sd, 4)


def _ollama_installed():
    try:
        return {m.get("model", "") for m in ollama.list().get("models", [])}
    except Exception:
        return set()


@app.get("/health")
def health():
    return {"status": "ok", "gpu": torch.cuda.is_available()}


@app.get("/models")
def api_models():
    """Allowlisted selector options for the Web UI, filtered to installed Ollama models, plus presets."""
    installed = _ollama_installed()

    def have(name):
        return name in installed or (":" not in name and f"{name}:latest" in installed)

    ollama_avail = [m for m in OLLAMA_MODELS if have(m)] or OLLAMA_MODELS
    return {
        "whisper": [WHISPER_LOCK] if WHISPER_LOCK else WHISPER_MODELS,
        "ollama": [REFINE_MODEL] if REFINE_MODEL else ollama_avail,
        "sd": [SD_LOCK] if SD_LOCK else list(SD_MODELS),
        "presets": {k: {"whisper": v["whisper"], "ollama": v["ollama"], "sd": v["sd"]}
                    for k, v in PRESETS.items()},
        "default_preset": DEFAULT_PRESET,
        "vram_estimate_mb": VRAM_ESTIMATE_MB,
        "locked": {"whisper": bool(WHISPER_LOCK), "ollama": bool(REFINE_MODEL), "sd": bool(SD_LOCK)},
    }


@app.get("/vram")
def api_vram():
    """Free/total GPU memory, for the UI pre-warning. Values are approximate on a shared GPU."""
    if not torch.cuda.is_available():
        return {"gpu": False}
    free, total = torch.cuda.mem_get_info()
    mb = 1024 * 1024
    return {"gpu": True, "free_mb": free // mb, "total_mb": total // mb}


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
    style: str = Form(""),
    whisper_model: str = Form(""),
    refine_model: str = Form(""),
    sd_model: str = Form(""),
):
    """Full pipeline: audio → transcribe → refine → generate → (optional) cast."""
    # Parse size
    try:
        width, height = map(int, size.split("x"))
    except ValueError:
        width, height = 768, 432

    # Get preset config, then resolve each stage's model (request field > lock > preset)
    config = PRESETS.get(preset, PRESETS[DEFAULT_PRESET])
    whisper_name, refiner_name, sd_name, sd_steps = _resolve_models(
        config, whisper_model.strip(), refine_model.strip(), sd_model.strip()
    )
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
    try:
        text = transcribe_audio(data, whisper_model=whisper_name)
    except torch.cuda.OutOfMemoryError:
        unload_models()
        raise HTTPException(503, f"transcription ran out of VRAM (whisper {whisper_name}); GPU may be busy")
    if not text:
        raise HTTPException(400, "No speech detected")

    # Free Whisper VRAM before loading diffusion model
    unload_whisper()

    # refine_prompt never raises; it degrades to the raw transcript on Ollama failure
    prompt = refine_prompt(text, ollama_model=refiner_name)

    # Append style suffix if provided
    if style:
        prompt = f"{prompt}, {style}"

    try:
        image = generate_image(prompt, width=width, height=height,
                               sd_model=sd_name, sd_steps=sd_steps)
    except torch.cuda.OutOfMemoryError:
        unload_models()
        raise HTTPException(503, f"image generation ran out of VRAM (diffusion {sd_name}); try a smaller model")

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
            "X-Models": f"{whisper_name} | {refiner_name} | {sd_name}",
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

    # For device_map models, remove accelerate hooks first
    for name, pipe in list(_sd_pipes.items()):
        try:
            # Remove accelerate dispatch hooks (holds tensor references)
            from accelerate.hooks import remove_hook_from_submodules
            remove_hook_from_submodules(pipe)
        except Exception:
            pass
        del pipe
    _sd_pipes.clear()

    for name, model in list(_whisper_models.items()):
        del model
    _whisper_models.clear()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return {"status": "models unloaded"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
