FROM python:3.10-slim-buster
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH="${PYTHONPATH}:/app/src"
CMD ["python", "main.py"]