FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Disable telemetry popup
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Launch application
CMD ["streamlit", "run", "langraphtoolfrontend.py", "--server.port=8501", "--server.address=0.0.0.0"]
