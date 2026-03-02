#!/bin/bash

CLUSTER_NAME=test-observability
TPU_TYPE=v6e-16
NUM_SLICES=1
ZONE=asia-northeast1-b

xpk workload create-pathways \
  --workload rl-hongmao \
  --num-slices=${NUM_SLICES} \
  --tpu-type=${TPU_TYPE} \
  --cluster=${CLUSTER_NAME} \
  --zone=${ZONE} \
  --docker-name='rl-hongmao-workload' \
  --docker-image='asia-northeast1-docker.pkg.dev/tpu-service-473302/sglang-project/sgl-jax:v0.1' \
  --command="JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 JAX_USE_SHARDY_PARTITIONER=0 python3 -u -m sgl_jax.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --trust-remote-code --tp-size=4 --mem-fraction-static=0.8 --chunked-prefill-size=2048 --download-dir=/tmp --dtype=bfloat16 --max-running-requests 8 --skip-server-warmup --page-size=64 --max-total-tokens=257536 --random-seed=27 --precompile-token-paddings=2048 --precompile-bs-paddings=8 --enable-single-process"
