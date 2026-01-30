#!/usr/bin/env python3
"""Web UI for Voice-to-Image: Browser mic recording → server API → image display."""

import io
import os

import httpx
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

# Server API URL (the FastAPI server)
API_URL = os.environ.get("API_URL", "http://localhost:8765")


@app.route("/")
def index():
    return render_template("index.html", api_url=API_URL)


@app.route("/api/pipeline", methods=["POST"])
def pipeline():
    """Proxy to the server API, handling the audio upload."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]
    cast = request.form.get("cast", "false") == "true"
    device = request.form.get("device", "")

    # Forward to server API
    try:
        files = {"audio": (audio_file.filename, audio_file.read(), audio_file.content_type)}
        data = {"cast": str(cast).lower(), "device": device}

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{API_URL}/pipeline", files=files, data=data)

        if resp.status_code != 200:
            return jsonify({"error": resp.text}), resp.status_code

        # Return image with metadata from headers
        return send_file(
            io.BytesIO(resp.content),
            mimetype="image/png",
            download_name="generated.png",
        ), 200, {
            "X-Transcription": resp.headers.get("X-Transcription", ""),
            "X-Prompt": resp.headers.get("X-Prompt", ""),
        }

    except httpx.ConnectError:
        return jsonify({"error": f"Cannot connect to server at {API_URL}"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    """Check server health."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{API_URL}/health")
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 503


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8766, debug=True)
