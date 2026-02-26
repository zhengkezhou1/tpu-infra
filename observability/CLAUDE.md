# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TPU observability stack: collects TPU hardware metrics from GKE-hosted TPU v6e nodes via `libtpu` SDK, exposes them as Prometheus metrics, and visualizes through Grafana backed by Google Managed Prometheus (GMP).

## Architecture

**Data flow:** libtpu SDK → `workload/metric.py` (Prometheus exporter on :8000) → GMP ClusterPodMonitoring scrape → GMP Frontend (PromQL proxy on :9090) → Grafana

- **workload/metric.py** - Prometheus exporter using `libtpu.sdk.tpumonitoring` to collect per-chip metrics (duty cycle, HBM usage, tensorcore utilization, HLO queue/timing)
- **workload/train.py** - JAX training workload with 2D mesh sharding (2x4 topology), used to generate TPU load for testing metrics
- **terraform/** - Full infrastructure: GKE cluster with TPU v6e-2x4 node pool (spot), GMP, Grafana via Helm, Workload Identity for both Grafana and GMP frontend
- **infra/** - SkyPilot task definitions for launching the cluster (`setup.yaml`) and running workloads (`metric_job.yaml`, `train_job.yaml`)
- **grafana/tpu.json** - Grafana dashboard definition

## Key Commands

```bash
# Infrastructure provisioning
cd terraform && terraform init && terraform apply -var="grafana_admin_password=XXX"

# Launch SkyPilot cluster (2-node TPU v6e)
sky launch -c dev infra/setup.yaml

# Run metrics exporter on TPU nodes
sky exec dev infra/metric_job.yaml

# Run training workload
sky exec dev infra/train_job.yaml

# Get Grafana IP
kubectl get svc grafana -n monitoring -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

## Key Details

- GCP project: `tpu-service-473302`, region: `asia-northeast1`
- TPU topology: v6e 2x4 (2 hosts, 4 chips each), machine type `ct6e-standard-4t`
- Terraform state backend: GCS bucket `tpu-service-terraform-state` prefix `infra/observability`
- SkyPilot jobs use `hostNetwork: true` and hardcoded DNS names (`dev-c65f19d1-head/worker1`)
- Multi-host JAX requires `JAX_COORDINATOR_ADDRESS`, `JAX_PROCESS_COUNT`, `JAX_PROCESS_ID` plus `TPU_WORKER_HOSTNAMES` (IPs without ports)
