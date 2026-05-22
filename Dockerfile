
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# create a non-root user for security alignment on Hugging Face
RUN useradd -m -u 1000 user
WORKDIR $HOME/app

# cp requirements and install dependencies as the non-root user
COPY --chown=user:user requirements.txt .
USER user
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the remaining application code and pre-built indices
# (Make sure chroma_db_index/ and bm25_index.pkl are in your root directory)
COPY --chown=user:user . .

# Inform Docker that the container listens on port 7860
EXPOSE 7860

# Configure Streamlit to run correctly inside the Hugging Face environment
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]