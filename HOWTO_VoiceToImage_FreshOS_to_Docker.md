# HOWTO: Fresh OS to Running Docker Container (Voice-To-Image)

Target: Ubuntu 24.04 ('noble') with an NVIDIA GPU, nothing installed yet. Ends with
`docker compose up -d` serving the Web UI at `http://localhost:8766`.

## 1. System update

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl git build-essential
```

## 2. NVIDIA driver

Check if a driver is already loaded:
```bash
nvidia-smi
```
If that fails (command not found or no devices), install the recommended driver:
```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```
After reboot, confirm:
```bash
nvidia-smi
```
Should show the GPU(s) and driver/CUDA version.

## 3. Docker Engine

Remove any distro packages that conflict, then install from Docker's official repo:
```bash
for pkg in docker.io docker-doc docker-compose podman-docker containerd runc; do
  sudo apt remove -y $pkg
done

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Let your user run docker without sudo (log out/in or `newgrp docker` to take effect):
```bash
sudo usermod -aG docker $USER
newgrp docker
```

Verify:
```bash
docker run --rm hello-world
```

## 4. NVIDIA Container Toolkit (GPU access inside containers)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify Docker can see the GPU:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 5. Clone the repo

```bash
git clone <this-repo-url> Voice-To-Image
cd Voice-To-Image
```

## 6. Ollama prerequisite (host-level, not containerized)

Ollama runs natively on the host, not in Docker — see the top comment in
`docker-compose.yml` for why. One idempotent script handles install + model pulls:
```bash
./scripts/setup-ollama.sh
```
This installs Ollama if missing, waits for `localhost:11434` to answer, and pulls
`llama3.2:1b`, `llama3.2`, and `qwen3.5:9b` (~9.9GB combined) if not already present.

## 7. Build and start the container

```bash
docker compose up -d --build
```

## 8. Verify

```bash
curl http://localhost:8765/health
# {"status": "ok", "gpu": true}
```
If `gpu` is `false`, re-check step 4.

Open the Web UI:
```
http://localhost:8766
```
(from another device on the LAN, use the host's IP instead of `localhost` — see the
README's **Microphone access** section for the Chrome/Firefox insecure-origin flag
needed for mic access over plain HTTP).

## 9. Optional: pre-download diffusion/Whisper models

Models lazy-download on first use per preset. To avoid first-request timeouts:
```bash
docker compose exec voice-to-image hf download stabilityai/sd-turbo
docker compose exec voice-to-image hf download stabilityai/sdxl-turbo
docker compose exec voice-to-image hf download Systran/faster-whisper-base
```
See the README's **Pre-download Models** section for the full list per preset.

## Useful commands

```bash
docker compose logs -f       # tail logs
docker compose down          # stop and remove container (weights persist on host)
docker compose up -d --build # rebuild after a code change
```
