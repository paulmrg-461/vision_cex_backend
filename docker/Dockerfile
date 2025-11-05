FROM nvidia/cuda:12.0-base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

COPY . /app

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "app.presentation.api.main:app", "--host", "0.0.0.0", "--port", "8000"]