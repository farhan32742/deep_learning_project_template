# 1. Base Image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install System Dependencies
# libgl1 is required for OpenCV/YOLO
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Install CPU-only PyTorch (Lightweight optimization)
# We install this BEFORE requirements to ensure we get the small wheels.
# We match the version from your requirements.txt (2.5.1)
RUN pip install --no-cache-dir --default-timeout=1000 torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 5. Copy the requirements file
COPY requirements.txt .


RUN sed -i '/torch/d; /mlflow/d; /dagshub/d; /dvc/d' requirements.txt && \
    pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 -r requirements.txt && \
    pip install --no-cache-dir python-multipart


COPY . .

ENV PYTHONPATH="/app/src"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]