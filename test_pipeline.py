#!/usr/bin/env python3
"""Test the voice-to-image pipeline with test.wav"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy import signal

SAMPLE_RATE = 16000

# Load and convert test.wav
print("Loading test.wav...")
sr, data = wavfile.read("test.wav")
print(f"Original: {sr}Hz, shape={data.shape}, dtype={data.dtype}")

# Convert to float32
if data.dtype == np.int16:
    data = data.astype(np.float32) / 32768.0

# Convert stereo to mono
if len(data.shape) > 1:
    data = data.mean(axis=1)

# Resample to 16kHz
if sr != SAMPLE_RATE:
    data = signal.resample(data, int(len(data) * SAMPLE_RATE / sr))
    print(f"Resampled to {SAMPLE_RATE}Hz, {len(data)} samples")

# Step 1: Transcribe
print("\n=== TRANSCRIBE ===")
from faster_whisper import WhisperModel
import tempfile

with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    wavfile.write(f.name, SAMPLE_RATE, (data * 32767).astype(np.int16))
    temp_path = f.name

model = WhisperModel("base", device="cuda" if torch.cuda.is_available() else "cpu")
segments, _ = model.transcribe(temp_path)
text = " ".join(seg.text for seg in segments).strip()
Path(temp_path).unlink()
print(f"Transcription: {text}")

if not text:
    print("No speech detected, exiting.")
    sys.exit(1)

# Step 2: Refine prompt
print("\n=== REFINE PROMPT ===")
import ollama

system = """You are a prompt engineer for image generation.
Convert the user's spoken description into a concise, vivid image prompt.
Focus on visual details: subject, style, lighting, colors, composition.
Output ONLY the prompt, nothing else. Keep it under 77 tokens."""

response = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ],
)
prompt = response["message"]["content"].strip()
print(f"Image prompt: {prompt}")

# Step 3: Generate image
print("\n=== GENERATE IMAGE ===")

from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
if torch.cuda.is_available():
    pipe = pipe.to("cuda")

image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
image.save("test_output.png")
print("Saved: test_output.png")

# Cleanup
del pipe
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n=== DONE ===")
