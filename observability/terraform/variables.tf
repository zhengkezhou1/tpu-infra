variable "project_id" {
  description = "The project ID to host the cluster in"
  type        = string
  default     = "tpu-service-473302"
}

variable "region" {
  description = "The region to host the cluster in"
  type        = string
  default     = "asia-northeast1"
}

variable "cluster_name" {
  description = "The name of the GKE cluster"
  type        = string
  default     = "test-observability"
}

variable "node_locations" {
  description = "The comma-separated list of one or more zones where GKE creates the node pool"
  type        = list(string)
  default     = ["asia-northeast1-b"]
}

variable "metrics_exporter_image" {
  description = "Docker image for the TPU metrics exporter"
  type        = string
  default     = "ghcr.io/zhengkezhou1/tpu-infra/metrics-exporter:sha-f196774"
}

variable "grafana_admin_password" {
  description = "Grafana admin password"
  type        = string
  sensitive   = true
}
