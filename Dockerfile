# ===============================
# Dockerfile
# ===============================

#changes in code for ON/OFF AWS toggle
# Use a modern, slim Python base
FROM python:3.12-slim

# Install system dependencies for LightGBM and performance
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements-dockers.txt .
RUN pip install --no-cache-dir -r requirements-dockers.txt

# Chromium for the single PDF pipeline (src/services/pdf_service.py):
# Export PDF and WhatsApp document delivery render the Next.js print route
# with headless Chromium via Playwright.
RUN playwright install --with-deps chromium

# Install DVC with S3 support (needed to pull model artifacts)
RUN pip install --no-cache-dir dvc[s3]

# Copy Full Project (including .dvc files)
COPY . .

# ---- Accept AWS credentials as build arguments ----
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION

# 🔥 USE SAME GLOBAL TOGGLE FROM GITHUB ACTIONS
ARG ENABLE_AWS=false

# Set them as environment variables for DVC/S3
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION

# Pull ML Artifacts ONLY when AWS is enabled
RUN if [ "$ENABLE_AWS" = "true" ]; then dvc pull; else echo "Skipping DVC pull (AWS disabled)"; fi

# Ensure 'src' and 'scripts' are discoverable by Python
ENV PYTHONPATH="/app"

# Expose FastAPI ports (8001 = public EstateMind Copilot API, 8000 = internal ML API)
EXPOSE 8001 8000

# Run both APIs: the ML Prediction API stays on 8000 (the Copilot API's
# prediction service calls it at 127.0.0.1:8000), and the EstateMind
# Copilot API on 8001 is the public entrypoint.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 & exec uvicorn src.api.main:app --host 0.0.0.0 --port 8001"]



