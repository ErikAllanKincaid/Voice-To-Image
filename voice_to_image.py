#!/usr/bin/env python3
"""Voice-to-Image: Record speech, transcribe, generate image, cast to Chromecast."""

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import ollama
import sounddevice as sd
import torch
import whisper
from diffusers import StableDiffusionPipeline
from PIL import Image
from scipy.io import wavfile

# Defaults
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
WHISPER_MODEL = "base"
OLLAMA_MODEL = "llama3.2"
SD_MODEL = "stabilityai/stable-diffusion-2-1"


def record_audio(duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Record audio from microphone."""
    print(f"Recording {duration}s...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("Recording complete.")
    return audio.flatten()


def record_push_to_talk(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Record while Space is held. Requires root for keyboard module."""
    import keyboard

    print("Hold SPACE to record...")
    keyboard.wait("space")
    print("Recording (release SPACE to stop)...")

    frames = []
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
        while keyboard.is_pressed("space"):
            data, _ = stream.read(int(sample_rate * 0.1))
            frames.append(data)

    print("Recording complete.")
    return np.concatenate(frames).flatten() if frames else np.array([])


def transcribe(audio: np.ndarray, model_name: str = WHISPER_MODEL) -> str:
    """Transcribe audio using Whisper."""
    print("Transcribing...")
    model = whisper.load_model(model_name)
    result = model.transcribe(audio, fp16=torch.cuda.is_available())
    text = result["text"].strip()
    print(f"Transcription: {text}")
    return text


def refine_prompt(text: str, model: str = OLLAMA_MODEL) -> str:
    """Use Ollama to convert speech to an image generation prompt."""
    print("Refining prompt...")

    system = """You are a prompt engineer for image generation.
Convert the user's spoken description into a concise, vivid image prompt.
Focus on visual details: subject, style, lighting, colors, composition.
Output ONLY the prompt, nothing else. Keep it under 77 tokens."""

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
    )
    prompt = response["message"]["content"].strip()
    print(f"Image prompt: {prompt}")
    return prompt


def generate_image(prompt: str, model_id: str = SD_MODEL) -> Image.Image:
    """Generate image using Stable Diffusion."""
    print("Generating image...")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

    # Cleanup
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Image generated.")
    return image


def cast_to_chromecast(image_path: Path, device: str | None = None):
    """Cast image to Chromecast using catt."""
    print(f"Casting to Chromecast{f' ({device})' if device else ''}...")
    cmd = ["catt"]
    if device:
        cmd.extend(["-d", device])
    cmd.extend(["cast", str(image_path)])
    subprocess.run(cmd, check=True)
    print("Cast complete.")


def run_pipeline(
    duration: float,
    push_to_talk: bool,
    device: str | None,
    whisper_model: str,
    ollama_model: str,
    sd_model: str,
    output_dir: Path | None,
) -> Path:
    """Run the full voice-to-image pipeline once."""
    # Record
    if push_to_talk:
        audio = record_push_to_talk()
    else:
        audio = record_audio(duration)

    if len(audio) < SAMPLE_RATE * 0.5:
        print("Audio too short, skipping.")
        return None

    # Transcribe
    text = transcribe(audio, whisper_model)
    if not text:
        print("No speech detected, skipping.")
        return None

    # Refine prompt
    prompt = refine_prompt(text, ollama_model)

    # Generate image
    image = generate_image(prompt, sd_model)

    # Save image
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / f"voice_image_{int(time.time())}.png"
    else:
        image_path = Path(tempfile.mktemp(suffix=".png"))
    image.save(image_path)
    print(f"Saved: {image_path}")

    # Cast
    if device is not None or device == "":
        cast_to_chromecast(image_path, device if device else None)

    return image_path


def main():
    parser = argparse.ArgumentParser(description="Voice-to-Image Chromecast")
    parser.add_argument("-d", "--device", help="Chromecast device name")
    parser.add_argument("-t", "--duration", type=float, default=RECORD_SECONDS, help="Recording duration (seconds)")
    parser.add_argument("-p", "--push-to-talk", action="store_true", help="Hold SPACE to record (requires root)")
    parser.add_argument("-c", "--continuous", action="store_true", help="Run continuously")
    parser.add_argument("-i", "--interval", type=float, default=30, help="Seconds between recordings in continuous mode")
    parser.add_argument("-o", "--output", type=Path, help="Output directory for images")
    parser.add_argument("--no-cast", action="store_true", help="Skip casting to Chromecast")
    parser.add_argument("--whisper-model", default=WHISPER_MODEL, help="Whisper model size")
    parser.add_argument("--ollama-model", default=OLLAMA_MODEL, help="Ollama model for prompt refinement")
    parser.add_argument("--sd-model", default=SD_MODEL, help="Stable Diffusion model")
    args = parser.parse_args()

    device = args.device if not args.no_cast else None

    if args.continuous:
        print(f"Continuous mode: recording every {args.interval}s. Ctrl+C to stop.")
        while True:
            try:
                run_pipeline(
                    args.duration, args.push_to_talk, device,
                    args.whisper_model, args.ollama_model, args.sd_model, args.output
                )
                if not args.push_to_talk:
                    print(f"Waiting {args.interval}s...")
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
    else:
        run_pipeline(
            args.duration, args.push_to_talk, device,
            args.whisper_model, args.ollama_model, args.sd_model, args.output
        )


if __name__ == "__main__":
    main()
