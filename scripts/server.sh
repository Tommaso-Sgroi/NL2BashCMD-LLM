#!/bin/bash
python3.10 -m llama_cpp.server --model './models/Meta-Llama-3-8B.Q8_0.gguf' \
  --n_gpu_layers -1 --seed 42 --verbose 1