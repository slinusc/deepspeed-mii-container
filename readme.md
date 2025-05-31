# DeepSpeed-MII OpenAI-Compatible Docker Container

This repository provides a **Dockerfile** and instructions to **pull a prebuilt image** from Docker Hub to run a Deepspeed-MII server to serve your locally deployed LLM. The container runs [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII) with an OpenAI-API-compatible endpoint. Once running, you can load any Hugging Face model (e.g., `mistralai/Mistral-7B-Instruct-v0.3`) or simply use the preloaded image on Docker Hub to interact via `/v1/chat/completions` just like you would with the official OpenAI API.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Repository Structure](#repository-structure)
3. [Option A: Pull the Prebuilt Image](#option-a-pull-the-prebuilt-image)
4. [Option B: Build from Source](#option-b-build-from-source)
5. [Running the Container](#running-the-container)
6. [Testing with Linux CLI (](#testing-with-linux-cli-curl)[`curl`](#testing-with-linux-cli-curl)[)](#testing-with-linux-cli-curl)
7. [Testing with Python + OpenAI SDK](#testing-with-python--openai-sdk)
8. [Environment Variables](#environment-variables)
9. [Customizing & Troubleshooting](#customizing--troubleshooting)
10. [License](#license)

---

## Prerequisites

* **Docker** (20.10+).
* **NVIDIA Container Toolkit** (to allow `--gpus all`).
* A valid **Hugging Face Hub token** if you plan to pull private or gated models:

  ```bash
  export HF_TOKEN=<your_hf_token>
  ```
* (Optional) **OpenAI Python SDK** for testing in Python:

  ```bash
  pip install openai
  ```

---

## Repository Structure

```
deepSpeed-mii-container/
├── Dockerfile
├── readme.md
└── .gitignore
```

* **`Dockerfile`**: Builds a CUDA-enabled image with DeepSpeed-MII, Pydantic v2, `pydantic-settings`, `sentencepiece`, FastAPI, Uvicorn, ShortUUID, and FastChat. Exposes port 23333 by default.
* **`readme.md`**: This file—contains instructions for pulling or building, running, and testing.
* **`.gitignore`**: Ignores local artifacts like `__pycache__` and logs.

---

## Option A: Pull the Prebuilt Image

If you want to skip building locally, simply pull the prebuilt image from Docker Hub:

```bash
# Pull the image (tagged "latest")
docker pull slinusc/deepspeed-mii:latest
```

Now you can jump to [Running the Container](#running-the-container) below, using `slinusc/deepspeed-mii:latest` as the image name.

---

## Running the Container

Whether you pulled the prebuilt image or built locally, run the container with GPU support, mounting your Hugging Face cache, and exposing port 23333:

```bash
# Using the prebuilt Docker Hub image:
docker run --runtime=nvidia --gpus all \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  -p 127.0.0.1:23333:23333 \
  --ipc=host \
  slinusc/deepspeed-mii:latest \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --port 23333
```

* `--runtime=nvidia --gpus all`: Access all GPUs.
* `-v $HOME/.cache/huggingface:/root/.cache/huggingface`: Mount HF cache so weights aren’t re-downloaded.
* `-e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN`: Pass HF token into container.
* `-p 127.0.0.1:23333:23333`: Map container port 23333 → host 23333.
* `--ipc=host`: Share IPC namespace to reduce overhead.
* `--model mistralai/Mistral-7B-Instruct-v0.3`: Hugging Face path of the model to load.
* `--port 23333`: Force Uvicorn to bind inside container on port 23333.

You should see logs like:

```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:23333 (Press CTRL+C to quit)
```

At that point, the server is live at `http://127.0.0.1:23333/v1/...`.

---

## Testing with Linux CLI (`curl`)

Once the container is running, open a new terminal and run:

### 1. Check available models

```bash
curl http://127.0.0.1:23333/v1/models
```

Expected JSON:

```json
{
  "object": "list",
  "data": [
    {
      "id": "mistralai/Mistral-7B-Instruct-v0.3",
      "object": "model",
      "created": 1748684820,
      "owned_by": "deepspeed-mii",
      "root": "mistralai/Mistral-7B-Instruct-v0.3",
      "parent": null,
      "permission": [ … ]
    }
  ]
}
```

The key field is `"id"`, which you must use for subsequent requests.

### 2. Send a chat completion request

```bash
curl http://127.0.0.1:23333/v1/chat/completions \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "messages": [
          { "role": "system", "content": "You are a helpful assistant." },
          { "role": "user",   "content": "Tell me a fun fact about penguins." }
        ],
        "max_tokens": 32,
        "temperature": 0.7
      }'
```

### 3. Send a text completion request (optional)

```bash
curl http://127.0.0.1:23333/v1/completions \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "prompt": "Once upon a time in a distant galaxy,",
        "max_tokens": 50,
        "temperature": 0.7
      }'
```

You’ll receive a JSON with a `choices` array containing the generated completion.

---

## Testing with Python + OpenAI SDK

Install the OpenAI SDK locally (if you haven’t already):

```bash
pip install openai
```

Save the following as `test_mii.py`:

```python
from openai import OpenAI

# Point to local MII endpoint:
client = OpenAI(
    api_key="",  # no key needed if container does not require auth
    base_url="http://127.0.0.1:23333/v1"
)

response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "How many wings does a penguin have?"}
    ],
    max_tokens=16,
    temperature=0.7
)
print(response.choices[0].message.content)
```

Run:

```bash
python3 test_mii.py
```

You should see a short answer about penguins printed. If it errors, ensure:

1. The container is still running.
2. The correct `model` ID is used.
3. Port 23333 is mapped.

---

## Environment Variables

* **`HF_TOKEN`** (or `HUGGING_FACE_HUB_TOKEN` inside the container)

  * Your Hugging Face Hub token for private gated models.

  ```bash
  export HF_TOKEN=<your_hf_token>
  ```

* **`OPENAI_API_KEY`** (optional)

  * If you configured the container to require an API key, set this on your host and pass it with `-e OPENAI_API_KEY` in `docker run`. Otherwise, the container defaults to no-auth mode.

---

## Customizing & Troubleshooting

### Change the CUDA base image

In `Dockerfile`, you can swap:

```dockerfile
FROM nvidia/cuda:12.2.2-devel-ubuntu20.04
```

for another tag such as `cuda:11.8-devel-ubuntu20.04` to match your GPU driver.

### Use a different Hugging Face model

Change the `--model` argument when running:

```bash
--model your-org/your-model-name
```

If you want to load a quantized checkpoint, append `--quantize gptq` or similar.

## License

This repository is licensed under the **MIT License**. See [LICENSE](LICENSE) for details. If you omit a LICENSE file, it defaults to “All rights reserved.”

---

**Congratulations!** You have a fully functional Docker container that runs DeepSpeed-MII in OpenAI-API compatibility mode. Anyone can now either pull the prebuilt image (`slinusc/deepspeed-mii:latest`) or build from source and run a local inference server for Mistral or any other Hugging Face model.
