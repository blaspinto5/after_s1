FROM python:3.11

WORKDIR /app

# Reemplazamos libgl1-mesa-glx por libgl1
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir opencv-python mediapipe

CMD ["python", "tracker.py"]
