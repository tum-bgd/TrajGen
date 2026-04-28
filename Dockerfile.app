# app/Dockerfile

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app/src/app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the application source and assets into the image
COPY pyproject.toml /app/
COPY src/ /app/src/

ENV PYTHONPATH=/app/src:/app/src/app

RUN python -m pip install --no-cache-dir \
    folium>=0.20.0 \
    matplotlib>=3.10.8 \
    osmnx>=2.0.7 \
    pandas>=2.3.3 \
    pydantic>=2.12.5 \
    pydantic-settings>=2.12.0 \
    pytest>=9.0.2 \
    scipy>=1.15.3 \
    shapely>=2.1.2 \
    streamlit>=1.54.0 \
    streamlit-folium>=0.26.1

EXPOSE 3972

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=3972", "--server.headless=true", "--browser.serverAddress=trajgen.api.bgd.ed.tum.de"]
