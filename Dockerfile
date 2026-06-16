#Dockerfile
# # Use a modern, slim Python base
# FROM python:3.12-slim

# # Install system dependencies for LightGBM and performance
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libgomp1 \
#     && rm -rf /var/lib/apt/lists/*

# # Set the working directory
# WORKDIR /app

# # Copy and install dependencies
# COPY requirements-dockers.txt ./
# RUN pip install --no-cache-dir -r requirements-dockers.txt

# # --- Project Structure ---
# # app.py resides in the root
# COPY app.py .
# COPY cache/run_information.json .

# # Copy internal packages/scripts
# COPY ./scripts/ ./scripts/
# COPY ./src/ ./src/
# COPY ./cache/ ./cache/

# # Ensure 'src' and 'scripts' are discoverable by Python
# ENV PYTHONPATH="/app"

# # Expose FastAPI port
# EXPOSE 8000

# # Run the API using uvicorn
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

#uncomment only this below code 
#------------------------------------------------------------------------------------------------------

# # Use a modern, slim Python base
# FROM python:3.12-slim

# # Install system dependencies for LightGBM and performance
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     git \
#     curl \
#     libgomp1 \
#     && rm -rf /var/lib/apt/lists/*

# # Set the working directory
# WORKDIR /app

# # Copy and install dependencies
# COPY requirements-dockers.txt .
# RUN pip install --no-cache-dir -r requirements-dockers.txt

# # Install DVC with S3 support (needed to pull model artifacts)
# RUN pip install --no-cache-dir dvc[s3]

# # Copy Full Project (including .dvc files)
# COPY . .

# # ---- Accept AWS credentials as build arguments ----
# ARG AWS_ACCESS_KEY_ID
# ARG AWS_SECRET_ACCESS_KEY
# ARG AWS_DEFAULT_REGION

# # Set them as environment variables for DVC/S3
# ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
# ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
# ENV AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION

# # Pull ML Artifacts from DVC Remote
# RUN dvc pull

# # Ensure 'src' and 'scripts' are discoverable by Python
# ENV PYTHONPATH="/app"

# # Expose FastAPI port
# EXPOSE 8000

# # Run the API using uvicorn
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

#------------------------------------------------------------------------------------------------------

# #changes in code for ON/OFF AWS toggle
# # Use a modern, slim Python base
# FROM python:3.12-slim

# # Install system dependencies for LightGBM and performance
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     git \
#     curl \
#     libgomp1 \
#     && rm -rf /var/lib/apt/lists/*

# # Set the working directory
# WORKDIR /app

# # Copy and install dependencies
# COPY requirements-dockers.txt .
# RUN pip install --no-cache-dir -r requirements-dockers.txt

# # Install DVC with S3 support (needed to pull model artifacts)
# RUN pip install --no-cache-dir dvc[s3]

# # Copy Full Project (including .dvc files)
# COPY . .

# # ---- Accept AWS credentials as build arguments ----
# ARG AWS_ACCESS_KEY_ID
# ARG AWS_SECRET_ACCESS_KEY
# ARG AWS_DEFAULT_REGION

# # 🔥 USE SAME GLOBAL TOGGLE FROM GITHUB ACTIONS
# ARG ENABLE_AWS=false

# # Set them as environment variables for DVC/S3
# ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
# ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
# ENV AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION

# # Pull ML Artifacts ONLY when AWS is enabled
# RUN if [ "$ENABLE_AWS" = "true" ]; then dvc pull; else echo "Skipping DVC pull (AWS disabled)"; fi

# # Ensure 'src' and 'scripts' are discoverable by Python
# ENV PYTHONPATH="/app"

# # Expose FastAPI port
# EXPOSE 8000

# # Run the API using uvicorn
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

#====================================================================================================================================================================

# ==========================================================
# MCP / FastAPI Docker Image
# ==========================================================

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN echo "===== VERIFY LIBGOMP ====="
RUN find / -name "libgomp.so*" 2>/dev/null || true
RUN ldconfig -p | grep gomp || true

WORKDIR /app

COPY requirements-dockers.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements-dockers.txt

COPY . .

ENV PYTHONPATH=/app

RUN ls -R /app/cache || true

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]