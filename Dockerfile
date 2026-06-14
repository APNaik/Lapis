FROM python:3.12-slim

# 1. Install system dependencies
# These are required for Docling, FAISS, and OpenCV (libGL)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Set working directory
WORKDIR /app

# 3. Install Python dependencies
# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Pre-download the Embedding Model
# This prevents the "No model found" error and speeds up container boot
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# 5. Copy the rest of the application code
COPY . .

# 6. Runtime setup
# Hugging Face Spaces and most container platforms expose the desired port
# through PORT. Default to 7860 so the app works out of the box on Spaces.
ENV PORT=7860
EXPOSE 7860

# 7. Execution command
# Use a shell form here so the PORT environment variable can be honored.
CMD ["sh", "-c", "streamlit run frontend.py --server.port ${PORT} --server.address 0.0.0.0"]