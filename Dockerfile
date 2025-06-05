############################################################
# DeepSpeed-MII • CUDA 11.8 • Ubuntu 22.04 • Python 3.10
############################################################
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# ----------  system + build tooling ----------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip python3-dev \       
        git ninja-build build-essential && \
    rm -rf /var/lib/apt/lists/*


# ----------  PyTorch + vision/audio (cu118 wheels) ----------
RUN pip3 install --no-cache-dir \
        torch==2.3.1+cu118 \
        torchvision==0.18.1+cu118 \
        torchaudio==2.3.1+cu118 \
        --index-url https://download.pytorch.org/whl/cu118


# ----------  DeepSpeed-MII and friends ----------
RUN pip3 install --no-cache-dir --upgrade \
        deepspeed-mii==0.3.3 \
        deepspeed-kernels \
        shortuuid uvicorn fastapi sentencepiece \
        "transformers>=4.40" \
        pydantic pydantic-settings

# ----------  sensible defaults ----------
ENV CUDA_HOME=/usr/local/cuda \
    DS_ACCELERATOR=cuda \
    DS_INFERENCE=1 \               
    DS_BUILD_OPS=0 \              
    PYTHONUNBUFFERED=1 \
    NCCL_P2P_DISABLE=1             

EXPOSE 23333

ENTRYPOINT ["python3", "-m", "mii.entrypoints.openai_api_server"]
CMD ["--host", "0.0.0.0", "--port", "23333"]
