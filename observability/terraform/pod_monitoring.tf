resource "kubernetes_manifest" "tpu_metrics_pod_monitoring" {
  manifest = yamldecode(<<-EOT
    apiVersion: monitoring.googleapis.com/v1
    kind: ClusterPodMonitoring # Use ClusterPodMonitoring for cross-namespace scraping
    metadata:
      name: tpu-metrics-exporter
      # The resource itself will be created in the 'monitoring' namespace by default if the provider is configured that way,
      # but the resource is cluster-scoped and doesn't have a `namespace` field in its own metadata.
    spec:
      selector:
        matchLabels:
          app: tpu-metrics-exporter
      endpoints:
      - port: "http" # Corresponds to the port named 'http' in config.yaml
        path: /
        interval: 30s
  EOT
  )
}
