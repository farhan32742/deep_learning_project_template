FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (needed for some python packages like cv2 or build tools)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Install the project itself (if not already handled by -e . in requirements, but good practice to ensure)
# The -e . in requirements.txt will handle this, but explicit pip install . is safer for non-dev builds.
# However, user requested -e . in requirements so we stick to that for dev consistency.

# Command to run (can be overridden)
CMD ["python", "main.py"]
