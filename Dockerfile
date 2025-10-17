# Dockerfile
# CUDA 12.1 + cuDNN 8 (PyTorch runtime)
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- system deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
      git git-lfs vim htop unzip tree \
      netcat \
  && rm -rf /var/lib/apt/lists/*
RUN git lfs install --system

# ---- create non-root user ----
ARG UID=1002
ARG GID=1002
RUN groupadd -g $GID appgroup && useradd -m -u $UID -g $GID appuser

WORKDIR /workspace
ENV HOME=/home/appuser

# ---- Python deps (non-root user) ----
USER appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"

# requirements.txt 복사 & 설치
COPY env/requirements.txt /workspace/env/requirements.txt
RUN pip install --no-cache-dir -r /workspace/env/requirements.txt

# 프로젝트 파일 복사
COPY --chown=appuser:appgroup . /workspace/
