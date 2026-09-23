# Voice-To-Image Docker Image
# Voice → Whisper → Ollama → Stable Diffusion → Chromecast
#
# Build: docker build -t voice-to-image .
# Run:   docker run --gpus all --network host voice-to-image
#
# Requires Ollama installed natively on the host and reachable at localhost:11434
# with llama3.2:1b, llama3.2, and qwen3.5:9b pulled -- run scripts/setup-ollama.sh once
# first. Not bundled in this image or in docker-compose.yml: see docker-compose.yml's
# top comment for why.

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
# (kept as an explicit pip list rather than `uv sync` — no PortAudio system lib is
# installed above, so sounddevice is intentionally omitted; mic capture is browser-side)
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    faster-whisper \
    diffusers \
    transformers \
    accelerate \
    catt \
    ollama \
    pillow \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    flask \
    httpx

# Copy application code
COPY server.py .
COPY webui/ webui/

# Startup script to run both services
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Model cache directory. Bind-mounted to the host's own ~/.cache/huggingface in
# docker-compose.yml, the same path the native (non-Docker) setup uses, so downloaded
# weights are shared between deployment methods and survive independent of the container.
ENV HF_HOME=/app/.cache/huggingface
VOLUME /app/.cache/huggingface

# Expose both ports
EXPOSE 8765 8766

# Run both server and webui
CMD ["./entrypoint.sh"]
