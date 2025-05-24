FROM python:3.9-slim

WORKDIR /app

# Install system dependencies OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    pkg-config \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libsm6 \
    libfontconfig1 \
    libice6 \
    libgl1-mesa-glx \
    libgtk-3-0 \
    curl \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install
RUN pip install --upgrade pip wheel

COPY requirements.txt .

# Install PyTorch and torchvision (CPU version - compatible versions)
RUN pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0

# Install Python dependencies
RUN pip install \
    streamlit \
    transformers \
    matplotlib \
    pillow \
    requests \
    timm==0.9.16 \
    opencv-python>=4.11.0.86 \
    scipy>=1.11.0 \
    numpy

# Install panopticapi
RUN pip install --no-cache-dir git+https://github.com/cocodataset/panopticapi.git

# Install detectron2
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/detectron2.git

# Copy application code
COPY . .

RUN mkdir -p static/sp

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "home.py", "--server.port=8501", "--server.address=0.0.0.0"]