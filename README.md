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
- Ollama running locally with llama3.2, llama3.2:1b, and qwen3.5:9b models
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
ollama pull llama3.2:1b   # Lite preset
ollama pull llama3.2      # Standard preset
ollama pull qwen3.5:9b    # High / Ultra / Flux presets
```

### Install dependencies
```bash
uv sync
```

## Running

Both servers must be running. The Web UI proxies requests to the API server.

**Terminal 1 — API server (port 8765):**
```bash
uv run python server.py
```

**Terminal 2 — Web UI (port 8766):**
```bash
uv run python webui/app.py
```

Access the Web UI at `http://localhost:8766`

## Pre-download Models

Models are lazy-loaded on first request, but you can pre-download to avoid timeouts.

### Whisper (faster-whisper, Hugging Face CLI)
```bash
uv run hf download Systran/faster-whisper-tiny      # Lite preset
uv run hf download Systran/faster-whisper-base      # Standard/High/Flux presets
uv run hf download Systran/faster-whisper-medium    # Ultra preset
uv run hf download Systran/faster-whisper-small     # available via per-stage selector
uv run hf download Systran/faster-whisper-large-v3  # available via per-stage selector
```

### LLM (Ollama)
```bash
ollama pull llama3.2:1b  # Lite preset
ollama pull llama3.2     # Standard preset
ollama pull qwen3.5:9b   # High / Ultra / Flux presets, ~7GB VRAM
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

## Web UI Notes

Access from any device on local network at `http://<server-ip>:8766`.

**Microphone access:** Browsers only expose the microphone API (`getUserMedia`) on secure contexts. If you see the error `Cannot read properties of undefined (reading 'getUserMedia')`, the page is being served over plain HTTP from a non-localhost origin.

- **Accessing via `localhost`:** Works without any changes. Use `http://localhost:8766`.
- **Accessing via LAN IP (e.g., from another device):** Chrome blocks mic access over plain HTTP. Workaround:
  1. Go to `chrome://flags/#unsafely-treat-insecure-origin-as-secure`
  2. Add your server URL (e.g., `http://192.168.1.43:8766`)
  3. Relaunch Chrome

Firefox has a similar setting at `about:config`, key `media.devices.insecure.enabled`.

## Quality Presets

| Preset | Whisper | LLM | Image Model | Steps | VRAM |
|--------|---------|-----|-------------|-------|------|
| Lite | tiny | llama3.2:1b | sd-turbo | 4 | ~4GB |
| Standard | base | llama3.2 | sd-turbo | 4 | ~6GB |
| High | base | qwen3.5:9b | sdxl-turbo | 4 | ~10GB |
| Ultra | medium | qwen3.5:9b | SDXL | 30 | ~24GB |
| Flux | base | qwen3.5:9b | FLUX.1-schnell | 4 | ~24GB* |

*FLUX automatically uses multi-GPU parallelism if available (2x 12GB works), or CPU offload on single GPU (slower).

## Per-Stage Model Selection

Beyond the presets above, the Web UI lets you override the Whisper, LLM, and diffusion
model independently via dropdowns (`GET /models` reports the allowlisted options, filtered
to Ollama models actually installed):

- **Whisper:** `tiny`, `base`, `small`, `medium`, `large-v3`
- **LLM (Ollama):** `llama3.2:1b`, `llama3.2`, `qwen3.5:9b`
- **Diffusion:** `sd-turbo`, `sdxl-turbo`, `stable-diffusion-xl-base-1.0`, `FLUX.1-schnell`

`/pipeline` rejects any model not on these allowlists, so a client cannot trigger an
arbitrary Hugging Face download on a shared host.

**Deployment locks:** set `WHISPER_MODEL`, `REFINE_MODEL`, or `SD_MODEL` env vars to pin
a stage to a single model and hide the alternative options in the Web UI. `REFINE_TEMPERATURE`
(default `0.2`) controls the LLM's sampling temperature during prompt refinement.

**Note on qwen3.5:** it emits a `<think>` reasoning block by default, which leaves
`message.content` empty for `ollama.chat` unless thinking is disabled. The server disables
it automatically for `qwen`/`deepseek`/`magistral`/`gpt-oss` model families; override with
the `REFINE_THINK` env var if needed.

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
- `GET /models` — Allowlisted per-stage model options, presets, and lock status
- `GET /vram` — Free/total GPU memory (approximate on a shared GPU)

## Chromecast

Requires `catt` (installed via dependencies). Set your device name in `server.py`:

```python
DEFAULT_CHROMECAST = "Living Room TV"
```

## Docker

Run Voice-to-Image in containers with GPU support. Includes Ollama sidecar.

> **Status: not yet run end-to-end.** These files build cleanly by inspection but have not
> been deployed to a real box. Before relying on them for a remote deploy, run through the
> checklist below on the target machine and fix anything that breaks.

### Deploy checklist (run on the target machine, once)
1. `nvidia-container-toolkit` installed and `docker run --rm --gpus all nvidia/smi ...` shows the GPU.
2. `docker compose up -d --build` — first run pulls the base image, builds the app image,
   and has the `ollama` sidecar pull `llama3.2:1b`, `llama3.2`, and `qwen3.5:9b` (~11GB total).
   This can take a long time on a slow link; watch `docker compose logs -f ollama` until all
   three pulls finish before testing presets that need them.
3. `curl http://localhost:8765/health` returns `{"status": "ok", "gpu": true}` — if `gpu` is
   `false`, the container cannot see the GPU; recheck step 1.
4. Open `http://<host>:8766`, run one generation per preset (Lite/Standard/High/Ultra/Flux)
   to confirm each pulled Ollama model and diffusion model actually loads.
5. `docker compose down` then `docker compose up -d` again — confirms the named volumes
   (`v2i-cache`, `ollama-models`) actually persisted the downloaded weights, so a restart
   does not re-download everything.
6. If serving to devices on the LAN rather than `localhost`, see **Microphone access** above —
   the Chrome flag / Firefox setting is still required over plain HTTP.

### Quick Start (docker compose)
```bash
# Requires nvidia-container-toolkit on host
docker compose up -d
# Access Web UI at http://localhost:8766
# API at http://localhost:8765
```

First run downloads models (Standard preset defaults):
- Whisper base: ~150MB
- SD-Turbo: ~3GB
- llama3.2: ~2GB
- qwen3.5:9b: ~7GB (only needed for High/Ultra/Flux presets)

### Manual Docker
```bash
# Build image
docker build -t voice-to-image .

# Run Ollama separately
docker run -d --gpus all -v ollama:/root/.ollama -p 11434:11434 ollama/ollama
docker exec -it <container> ollama pull llama3.2
docker exec -it <container> ollama pull qwen3.5:9b  # for High/Ultra/Flux presets

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

