# Voice-to-Image

Speak a description, get an AI-generated image. Optionally cast to Chromecast.

## Pipeline

1. **Record** — Browser microphone via Web Audio API
2. **Transcribe** — faster-whisper (speech-to-text)
3. **Refine** — Ollama LLM converts speech to image prompt
4. **Generate** — Stable Diffusion creates the image
5. **Cast** — (optional) Display on Chromecast via catt

## Requirements

- Python 3.12+
- CUDA GPU (16GB+ VRAM recommended for high quality mode)
- Ollama running locally with llama3.2 model
- uv package manager

## Setup

### Install uv (Python package manager)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
See [uv documentation](https://docs.astral.sh/uv/) for other install methods.

### Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
```

### Install dependencies
```bash
uv sync

# Run the server
uv run python server.py
```

Server runs on port 8765.

## Pre-download Models

Models are lazy-loaded on first request, but you can pre-download to avoid timeouts.

### LLM (Ollama)
```bash
ollama pull llama3.2
ollama pull llama3.2:1b  # for Lite preset
```

### Diffusion Models (Hugging Face CLI)
```bash
uv pip install huggingface_hub

uv run hf download stabilityai/sd-turbo        # Lite/Standard, ~5GB
uv run hf download stabilityai/sdxl-turbo      # High preset, ~13GB
uv run hf download stabilityai/stable-diffusion-xl-base-1.0  # Ultra, ~13GB (requires login)
```

For authenticated models:
```bash
uv run hf login
uv run hf download stabilityai/stable-diffusion-xl-base-1.0
```

Models cache to `~/.cache/huggingface/hub/`.

## Web UI

```bash
uv run python webui/app.py
```

Web UI runs on port 8766. Access from any device on local network.

**Note:** For remote mic access (not localhost), Chrome requires HTTPS or a flag:
```
chrome://flags/#unsafely-treat-insecure-origin-as-secure
```
Add your server URL (e.g., `http://192.168.1.43:8766`) and restart Chrome.

## Quality Presets

| Preset | Whisper | LLM | Image Model | Steps | VRAM |
|--------|---------|-----|-------------|-------|------|
| Lite | tiny | llama3.2:1b | sd-turbo | 4 | ~4GB |
| Standard | base | llama3.2 | sd-turbo | 4 | ~6GB |
| High | base | llama3.2 | sdxl-turbo | 4 | ~10GB |
| Ultra | medium | llama3.2 | SDXL | 30 | ~24GB |
| Flux | base | llama3.2 | FLUX.1-schnell | 4 | ~24GB* |

*FLUX automatically uses multi-GPU parallelism if available (2x 12GB works), or CPU offload on single GPU (slower).

## Image Sizes

All 16:9 aspect ratio:
- 640x360 (Lite)
- 768x432 (Standard)
- 1024x576 (HD)
- 1280x720 (720p)

## API Endpoints

- `POST /pipeline` — Full pipeline (audio file → image)
- `POST /transcribe` — Audio → text
- `POST /refine` — Text → image prompt
- `POST /generate` — Prompt → image
- `POST /cast` — Cast image to Chromecast
- `POST /unload` — Free GPU memory
- `GET /health` — Status check

## Chromecast

Requires `catt` (installed via dependencies). Set your device name in `server.py`:

```python
DEFAULT_CHROMECAST = "Living Room TV"
```

## Docker

Run Voice-to-Image in containers with GPU support. Includes Ollama sidecar.

### Quick Start (docker compose)
```bash
# Requires nvidia-container-toolkit on host
docker compose up -d
# Access Web UI at http://localhost:8766
# API at http://localhost:8765
```

First run downloads models:
- Whisper base: ~150MB
- SD-Turbo: ~3GB
- llama3.2: ~2GB

### Manual Docker
```bash
# Build image
docker build -t voice-to-image .

# Run Ollama separately
docker run -d --gpus all -v ollama:/root/.ollama -p 11434:11434 ollama/ollama
docker exec -it <container> ollama pull llama3.2

# Run Voice-to-Image
docker run --gpus all -p 8765:8765 -p 8766:8766 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -v v2i-cache:/app/.cache \
  voice-to-image
```

### Kubernetes
```bash
kubectl apply -f k8s-deployment.yaml
```

Includes:
- Ollama deployment with PVC for models
- Voice-to-Image deployment with GPU request
- Services for API (8765) and WebUI (80)

---

