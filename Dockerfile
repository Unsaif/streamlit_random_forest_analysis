FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install the library (and the bundled sample dataset) first so the layer is cached.
COPY pyproject.toml README.md LICENSE ./
COPY fastforest ./fastforest
RUN pip install ".[app]"

COPY app ./app
COPY .streamlit ./.streamlit

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --retries=5 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.headless", "true", "--server.port", "8501", "--server.address", "0.0.0.0"]
