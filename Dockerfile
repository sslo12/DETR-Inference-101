FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required for compilation and OpenCV
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
    libxext6 \
    libfontconfig1 \
    libice6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade pip wheel

COPY requirements.txt .

# Install PyTorch and torchvision (CPU version - change if you need GPU)
RUN pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0

# Install Python dependencies except detectron2
RUN pip install --no-cache-dir \
    streamlit \
    transformers \
    matplotlib \
    pillow \
    requests \
    timm==0.9.16 \
    opencv-python-headless>=4.8.0 \
    scipy>=1.11.0

# Install panopticapi
RUN pip install --no-cache-dir git+https://github.com/cocodataset/panopticapi.git

# Install detectron2
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/detectron2.git

COPY . .

RUN mkdir -p static/sp

EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# run the application
CMD ["streamlit", "run", "home.py", "--server.port=8501", "--server.address=0.0.0.0"]