# Use official Python 3.8 slim image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install basic system dependencies (optional but recommended)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy your project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default Streamlit port
EXPOSE 8501

# Command to run your app
CMD ["streamlit", "run", "main.py", "--server.headless", "true", "--server.port", "8501", "--server.enableCORS", "false"]
