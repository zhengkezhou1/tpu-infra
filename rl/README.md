<!-- ```zsh
PROJECT_ID=tpu-service-473302
CLUSTER_NAME=test-observability
TPU_TYPE=v6e-8
COMPUTE_ZONE=asia-northeast1-b
NUM_SLICES=2

xpk cluster adapt \
  --cluster=$CLUSTER_NAME --tpu-type=$TPU_TYPE \
  --zone=$COMPUTE_ZONE  --project=$PROJECT_ID \
  --num-slices=$NUM_SLICES \
  --skip-validation
```

```zsh
PROJECT_ID=tpu-service-473302
CLUSTER_NAME=test-observability
TPU_TYPE=v6e-16
COMPUTE_ZONE=asia-northeast1-b
NUM_SLICES=1

xpk cluster create \
  --cluster=$CLUSTER_NAME --tpu-type=$TPU_TYPE \
  --cpu-limit=12 --memory-limit=400Gi \
  --zone=$COMPUTE_ZONE  --project=$PROJECT_ID \
  --num-slices=$NUM_SLICES \
  --spot
``` -->

# RL
## Cluster
```zsh
export PROJECT_ID=tpu-service-473302
export CLUSTER_NAME=test-observability
export TPU_TYPE=v6e-16
export NUM_SLICES=1
export ZONE=asia-northeast1-b

xpk cluster create-pathways \
--cluster $CLUSTER_NAME \
--num-slices=$NUM_SLICES \
--tpu-type=$TPU_TYPE \
--zone=$ZONE \
--spot
```

## Storage
[FUSE](https://github.com/zhengkezhou1/xpk/blob/main/docs/usage/storage.md)

```zsh
export PROJECT_ID=tpu-service-473302
export CLUSTER_NAME=test-observability
export ZONE=asia-northeast1-b
xpk storage attach test-fuse-storage --type=gcsfuse \
    --project=$PROJECT_ID --cluster=$CLUSTER_NAME --zone=$ZONE \
    --mount-point='/pathways_rl_tmp' --readonly=false \
    --bucket=pathways_rl_tmp --size=1 --auto-mount=false
```

## Workload
[Workload](https://github.com/zhengkezhou1/xpk/blob/main/docs/usage/workloads.md)

```zsh
docker build --platform linux/amd64 -t ghcr.io/zhengkezhou1/sglang-project/sgl-jax:v0.1 .
```

```zsh
export CLUSTER_NAME=test-observability
export TPU_TYPE=v6e-16
export NUM_SLICES=1
export ZONE=asia-northeast1-b

xpk workload create-pathways \
  --workload rl-hongmao \
  --num-slices=$NUM_SLICES \
  --tpu-type=$TPU_TYPE \
  --cluster=$CLUSTER_NAME \
  --zone=$ZONE \
  --docker-name='rl-hongmao-workload' \
  --docker-image='asia-northeast1-docker.pkg.dev/tpu-service-473302/sg
lang-project/sgl-jax:v0.1' \
  --command="JAX_PLATFORMS=proxy
JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 JAX_USE_SHARDY_PARTITIONER=0
  python3 -u -m sgl_jax.launch_server --model-path
deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --trust-remote-code
--tp-size=4 --mem-fraction-static=0.8 --chunked-prefill-size=2048
--download-dir=/tmp --dtype=bfloat16 --max-running-requests 8
--skip-server-warmup --page-size=64 --max-total-tokens=257536
--random-seed=27 --precompile-token-paddings=2048
--precompile-bs-paddings=8 --enable-single-process"
```

[pathways-utils](https://github.com/AI-Hypercomputer/pathways-utils)