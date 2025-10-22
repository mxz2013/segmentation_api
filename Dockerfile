# --- STAGE 1: BUILD (Dependency Installation) ---
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

# Set up environment variables (fixed format)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python 3.10 (native to Ubuntu 22.04)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
# Install PyTorch with CUDA support from PyTorch index
# Then install other requirements
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.9.0+cu126 torchvision && \
    pip install --no-cache-dir -r requirements.txt

# --- STAGE 2: RUNTIME (Final Leaner Image) ---
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Re-install Python 3.10 runtime components
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Copy the rest of the application code
COPY . .

# Expose the API port
EXPOSE 8000

# Set environment variables for runtime
ENV HOST=0.0.0.0
ENV PORT=8000
ENV WORKERS=1
ENV MODEL=google/deeplabv3_mobilenet_v2_1.0_513
ENV DEVICE=cpu

# Command to start the server with arguments from environment variables
CMD python3 server_main.py \
    --host ${HOST} \
    --port ${PORT} \
    --workers ${WORKERS} \
    --model ${MODEL} \
    --device ${DEVICE} \
