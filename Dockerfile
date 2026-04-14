# Use a devel image that supports newer CUDA versions
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    curl \
    git \
    build-essential \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip and install Torch with the specific index-url for cu129
RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# Copy requirement files
COPY requirements.txt .
# Install web dependencies
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    facenet-pytorch \
    -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p web/uploads web/outputs

# Expose port 8000
EXPOSE 8000

# Run the backend server
CMD ["python3", "web/backend/main.py"]