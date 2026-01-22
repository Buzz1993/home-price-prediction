# Use a modern, slim Python base
FROM python:3.12-slim

# Install system dependencies for LightGBM and performance
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements-dockers.txt ./
RUN pip install --no-cache-dir -r requirements-dockers.txt

# --- Project Structure ---
# app.py resides in the root
COPY app.py .
COPY run_information.json .

# Copy internal packages/scripts
COPY ./scripts/ ./scripts/
COPY ./src/ ./src/
COPY ./cache/ ./cache/

# Ensure 'src' and 'scripts' are discoverable by Python
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Expose FastAPI port
EXPOSE 8000

# Run the API using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]