#!/bin/bash
docker run \
  --rm  \
  -p 8080:8080 -p 1337:1337 -p 7900:7900 \
  --shm-size="2g" \
  hlohaus789/g4f:latest
