FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf \
    PORT=7860

WORKDIR /app

# CPU-only torch keeps the image ~1.5 GB smaller than the CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# MODEL_ID can be a local folder (./model) or a Hub repo id
ENV MODEL_ID=./model
EXPOSE 7860

# One worker: each worker loads its own 2+ GB copy of the model
CMD gunicorn app:app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 180 --preload
