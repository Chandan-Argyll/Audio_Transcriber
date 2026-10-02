FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /app/data
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Default worker: CPU-first Parakeet + sherpa-onnx diarization, no PyTorch.
FROM base AS cpu
RUN pip install --no-cache-dir -r requirements-cpu.txt
CMD ["python", "-m", "worker.main"]

# Optional: adds pyannote Community-1 (CPU PyTorch) to the CPU worker.
FROM cpu AS pyannote
RUN pip install --no-cache-dir -r requirements-pyannote.txt

# Optional: WhisperX large-v3 + pyannote on CPU.
FROM base AS models
COPY requirements-models.txt .
RUN pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements-models.txt

FROM base AS evaluation
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements-eval.txt
ENTRYPOINT ["python", "-m", "eval.evaluate"]
