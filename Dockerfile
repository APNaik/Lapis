FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt $HOME/app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# FIXED: Combined the copy command into one line
COPY --chown=user . $HOME/app

RUN mkdir -p $HOME/app/vector_db

CMD ["streamlit", "run", "frontend.py", "--server.port=7860", "--server.address=0.0.0.0"]