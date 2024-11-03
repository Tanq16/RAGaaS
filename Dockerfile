FROM ubuntu:jammy

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y \
    python3 python3-pip python3-dev python3-venv git \
    wget curl unzip libmagic1 \
    && apt-get clean

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install langchain langchain_community langchain_ollama langchain_chroma langchain-qdrant qdrant-client

RUN mkdir /app
WORKDIR /app

COPY . .

ENTRYPOINT [ "python3", "main.py" ]
