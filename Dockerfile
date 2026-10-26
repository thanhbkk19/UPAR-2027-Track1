# Environment used for the reported results: Python 3.12, PyTorch 2.11 (CUDA 12.8 wheels), timm 1.0.22.
# Needs a host NVIDIA driver (>= 570) and the NVIDIA container toolkit (docker run --gpus all).
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR /workspace/UPAR2027_Track1

RUN pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["bash"]
