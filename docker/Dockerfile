FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libpoppler-cpp-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.8.2

COPY pyproject.toml poetry.lock* ./

# CPU-only PyTorch (~200Mo vs ~2.5Go avec CUDA)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

COPY . .

RUN pip install --no-cache-dir -e .

EXPOSE 8080

CMD ["chainlit", "run", "app/chainlit_app.py", "--host", "0.0.0.0", "--port", "8080"]
