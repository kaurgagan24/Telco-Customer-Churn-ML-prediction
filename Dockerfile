
# 1. Choose a Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependency required by XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Prevent Python from creating unnecessary cache files
ENV PYTHONDONTWRITEBYTECODE=1

# 5. Display Python logs immediately
ENV PYTHONUNBUFFERED=1

# 6. Copy dependency list
COPY requirements-docker.txt .

# 7. Install Python dependencies
RUN pip install --no-cache-dir -r requirements-docker.txt

# 8. Copy application and source code
COPY app/ ./app/
COPY src/ ./src/

# 9. Copy configuration and trained model artifacts
COPY configs/ ./configs/
COPY artifacts/ ./artifacts/

# 10. Make API port available
EXPOSE 8000

# 11. Start FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
