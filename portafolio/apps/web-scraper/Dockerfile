FROM python:3.11

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Instalar Scrapy y Playwright
RUN pip install --no-cache-dir scrapy playwright scrapy-playwright

# Instalar Chromium (Playwright necesita navegador)
RUN playwright install --with-deps chromium

# Copiar proyecto
COPY . .

CMD ["scrapy", "crawl", "ejemplo"]
