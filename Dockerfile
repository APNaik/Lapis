FROM python:3.11-slim

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

# 6. Render setup
# Render provides a PORT environment variable, but Streamlit defaults to 8501.
# We will use 8501 and map it in the Render Dashboard.
EXPOSE 8501

# 7. Execution Command
# Using the standard Streamlit start command.
# Note: We are not using a shell script here to keep it simple, 
# as Render handles networking natively.
CMD ["streamlit", "run", "frontend.py", "--server.port", "8501", "--server.address", "0.0.0.0"]