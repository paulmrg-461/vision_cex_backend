FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/backend

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3 \
       python3-pip \
       git \
       libgl1 \
       libglib2.0-0 \
       ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt \
    && pip3 install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

COPY . /app

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "app.presentation.api.main:app", "--host", "0.0.0.0", "--port", "8000"]