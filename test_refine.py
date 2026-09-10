#!/usr/bin/env python3
"""Manual test: run transcribed speech through refine_prompt for each candidate model.

No GPU needed. Requires Ollama running with the models pulled.

  .venv/bin/python test_refine.py                     # default model set
  .venv/bin/python test_refine.py qwen3.5:9b-q8_0     # specific model(s)
"""
import os
import sys

from server import refine_prompt

# Transcripts from the test_*.wav files (Whisper base).
TRANSCRIPTS = {
    "frog-raven": "funny frog on a lily pad talking to a raven they were both talking and they have hats on",
    "robot-spaceship": "I love robots and spaceships.",
    "electronics-room": (
        "The room has a long shelf with a lot of equipment on it. In the middle of the room "
        "there is a big table for a lot of people to do glasses or anything else they want. On "
        "the wall there is a very large screen for maybe putting up some videos or something. On "
        "the other side of the room there is a whole bunch of components. The room is called the "
        "electronics room."
    ),
}

MODELS = sys.argv[1:] or ["llama3.2:1b", "llama3.2:latest", "qwen3.5:9b-q8_0"]

if os.environ.get("REFINE_MODEL"):
    print(f"WARNING: REFINE_MODEL={os.environ['REFINE_MODEL']!r} overrides the per-model loop\n")

for model in MODELS:
    print(f"\n{'=' * 72}\nMODEL: {model}\n{'=' * 72}")
    for name, text in TRANSCRIPTS.items():
        out = refine_prompt(text, ollama_model=model)
        tag = "  [FALLBACK: output equals transcript]" if out.strip() == text.strip() else ""
        print(f"\n[{name}]{tag}\n  IN : {text}\n  OUT: {out}")
