# DeepSpeed-MII OpenAI-Compatible Docker Container

This repository provides a Dockerfile to build a GPU-enabled container that runs [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII) with an OpenAI-API-compatible endpoint. With this container, you can load any Hugging Face model (e.g., `mistralai/Mistral-7B-Instruct-v0.3`) and interact with it via `/v1/chat/completions` just like you would with the official OpenAI API.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Repository Structure](#repository-structure)
3. [Building the Docker Image](#building-the-docker-image)
4. [Running the Container](#running-the-container)
5. [Testing with Linux CLI (`curl`)](#testing-with-linux-cli-curl)
6. [Testing with Python + OpenAI SDK](#testing-with-python--openai-sdk)
7. [Environment Variables](#environment-variables)
8. [Customizing & Troubleshooting](#customizing--troubleshooting)
9. [License](#license)

---

## Prerequisites

Before you begin, make sure you have:

* **Docker** (version 20.10 or later) installed on your machine.

* **NVIDIA Container Toolkit** (nvidia-docker2) properly installed so that `--gpus all` works.

* A valid **Hugging Face Hub token** (if you plan to pull private or gated models).

  ```bash
  export HF_TOKEN=<your_hf_token>
  ```

  You will pass this token into the container so that it can download or authenticate to private repositories on Hugging Face.

* (Optional) The **OpenAI Python SDK** if you plan to test via Python:

  ```bash
  pip install openai
  ```

---

## Repository Structure

```
deepSpeed-mii-container/
├── Dockerfile
├── README.md
└── .gitignore
```

* **`Dockerfile`**
  Builds an image based on `nvidia/cuda:12.2.2-devel-ubuntu20.04`, installs Python3, DeepSpeed-MII, Pydantic v2, `pydantic-settings`, `sentencepiece`, FastAPI, Uvicorn, ShortUUID, and FastChat. It exposes port 23333 and configures the OpenAI-API entrypoint.

* **`README.md`**
  This file—contains build/run instructions, Linux/CURL examples, and Python/OpenAI-SDK examples.

* **`.gitignore`** *(optional)*
  Ignore files you don’t want in Git (e.g., `__pycache__/`, `*.log`, local build artifacts).

---

## Building the Docker Image

1. **Clone or copy this repository** to your local machine:

   ```bash
   git clone https://github.com/<your-username>/deepspeed-mii-container.git
   cd deepspeed-mii-container
   ```

2. **Build the Docker image** and tag it `deepspeed-mii-openai:latest`:

   ```bash
   docker build -t deepspeed-mii-openai:latest .
   ```

   The build steps will:

   1. Pull `nvidia/cuda:12.2.2-devel-ubuntu20.04` (with `nvcc` included).
   2. Install system packages: `python3`, `python3-dev`, `build-essential`, `ninja-build`, and `git`.
   3. Install the latest DeepSpeed-MII from GitHub.
   4. Install Python packages:

      * `pydantic` (v2.x)
      * `pydantic-settings`
      * `sentencepiece`
      * `fastapi`, `uvicorn`, `shortuuid`, `fastchat`

   **Note**: The final image will be several GB in size because of CUDA, PyTorch, DeepSpeed, and model dependencies.

---

## Running the Container

Once the image is built, you can run the container and load a specific Hugging Face model. The example below loads `mistralai/Mistral-7B-Instruct-v0.3` and exposes port 23333 on localhost.

```bash
docker run --runtime=nvidia --gpus all \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  -p 127.0.0.1:23333:23333 \
  --ipc=host \
  deepspeed-mii-openai:latest \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --port 23333
```

* `--runtime=nvidia --gpus all`
  Grants the container access to all available NVIDIA GPUs.

* `-v $HOME/.cache/huggingface:/root/.cache/huggingface`
  Mounts your Hugging Face cache directory so that model weights aren’t re-downloaded on each run.

* `-e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN`
  Passes your HF token into the container for private/private-gated model access.

* `-p 127.0.0.1:23333:23333`
  Maps the container’s port 23333 → host’s port 23333.

* `--ipc=host`
  Shares the host IPC namespace for slightly lower latency when loading large model files.

* `deepspeed-mii-openai:latest`
  The Docker image you just built.

* `--model mistralai/Mistral-7B-Instruct-v0.3`
  Specifies which Hugging Face model to load at startup.

* `--port 23333`
  Tells Uvicorn to bind to port 23333 inside the container (matching `-p 23333:23333`).

---

After a few seconds, you should see logs similar to:

```
INFO:     Started server process [XXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:23333 (Press CTRL+C to quit)
```

At that point, the server is live at `http://127.0.0.1:23333/v1/...`.

---

## Testing with Linux CLI (`curl`)

Once the container is running and you see “Uvicorn running on [http://0.0.0.0:23333,”](http://0.0.0.0:23333,”) open a new terminal on your host and use `curl` to verify the endpoints.

### 1. Check available models

```bash
curl http://127.0.0.1:23333/v1/models
```

A successful response looks like:

```json
{
  "data": [
    {
      "id": "mii",
      "object": "model",
      "owned_by": "deepspeed-mii"
    }
  ]
}
```

### 2. Send a chat/completion request

```bash
curl http://127.0.0.1:23333/v1/chat/completions \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
        "model": "mii",
        "messages": [
          { "role": "system",  "content": "You are a helpful assistant." },
          { "role": "user",    "content": "Tell me a fun fact about penguins." }
        ],
        "max_tokens": 32
      }'
```

You should receive a JSON response similar to:

```json
{
  "id": "chatcmpl-XXXXX",
  "object": "chat.completion",
  "created": 1717100000,
  "model": "mii",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Did you know that penguins can drink seawater? They have specialized glands that filter out salt!"
      },
      "finish_reason": "stop"
    }
  ]
}
```

---

## Testing with Python + OpenAI SDK

If you prefer to use Python code (for more complex workflows or scripting), install the official OpenAI Python SDK on your host:

```bash
pip install openai
```

Save the following to a file (e.g. `test_mii.py`):

```python
import os
from openai import OpenAI

# Point the OpenAI client at your local MII endpoint
# (no API key is required unless you configured one in the container)
client = OpenAI(
    api_key="",  # leave blank if no auth is configured
    base_url="http://127.0.0.1:23333/v1"
)

# Send a chat completion request
response = client.chat.completions.create(
    model="mii",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "How many wings does a penguin have?"}
    ],
    max_tokens=16
)

# Print out the assistant’s reply
print(response.choices[0].message.content)
```

Run the script:

```bash
python3 test_mii.py
```

You should see a short fact about penguins printed. If it stalls or errors, verify that:

1. The container is still running.
2. You used `--port 23333` when starting it.
3. No firewall or port conflict prevented binding to 23333.

---

## Environment Variables

* `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN` inside the container)
  Your Hugging Face Hub token. Used to authenticate downloads from private/premium repos.

  ```bash
  export HF_TOKEN=<your_hf_token>
  ```

* `OPENAI_API_KEY` (optional)
  If you configured your container’s OpenAI-compatible endpoint to require an API key, set `OPENAI_API_KEY` on your host and pass it into the container as `-e OPENAI_API_KEY=$OPENAI_API_KEY`. Otherwise, MII defaults to no-auth mode.

---

## Customizing & Troubleshooting

### Change the CUDA base image

In `Dockerfile`, replace:

```dockerfile
FROM nvidia/cuda:12.2.2-devel-ubuntu20.04
```

with any other CUDA “devel” tag that matches your GPU driver compatibility (e.g., `cuda:11.8-devel-ubuntu20.04`).

### Use a different Hugging Face model

When you run the container, change:

```bash
--model mistralai/Mistral-7B-Instruct-v0.3
```

to any supported HF model ID. For quantized weights, you can also pass:

```bash
--quantize gptq
```

(for example) to load a GPTQ-quantized checkpoint.

### “df: /root/.triton/autotune: No such file or directory”

You may see:

```
df: /root/.triton/autotune: No such file or directory
```

This is harmless. Triton (used by DeepSpeed inference kernels) will create `~/.triton/autotune` automatically if/when it needs to store autotuning data. You can ignore this warning.

### GPU OOM or Missing Headers

* If you see “CUDA out of memory,” your GPU does not have enough VRAM to load the full fp16 Mistral 7B. Consider using a smaller model or a quantized checkpoint.
* If you see a “Python.h: No such file or directory” error while building, ensure you installed `python3-dev` and `ninja-build` in the Dockerfile.

### Binding to a Different Host Interface

By default, the container binds to `0.0.0.0:23333`. If you only want to listen on `localhost`, keep `-p 127.0.0.1:23333:23333`. If you want external access (for remote testing), use:

```bash
-p 23333:23333
```

instead of `127.0.0.1:23333:23333`. Be aware of security implications if you expose it publicly.

---

## License

This repository is licensed under the **MIT License**. See [LICENSE](LICENSE) (create a LICENSE file with the full MIT text) for details. If you omit a LICENSE file, the default is “All rights reserved.”

---

**Congratulations!** You now have a self-contained Docker container that runs DeepSpeed-MII in OpenAI-API compatibility mode. Consumers can `git clone`, `docker build`, and `docker run` to spin up a local inference server for any Hugging Face model of their choice.
