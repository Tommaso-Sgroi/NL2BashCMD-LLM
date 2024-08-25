#!/bin/bash
cd ./ollama-main/ollama;
docker build --build-arg TARGETARCH=amd64 -f Dockerfile -t ollama-final-amd64 .
cd ../../.