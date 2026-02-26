resource "kubernetes_daemon_set_v1" "tpu_metrics_exporter" {
  metadata {
    name      = "tpu-metrics-exporter"
    namespace = "default"
    labels = {
      app = "tpu-metrics-exporter"
    }
  }

  spec {
    selector {
      match_labels = {
        app = "tpu-metrics-exporter"
      }
    }

    template {
      metadata {
        labels = {
          app = "tpu-metrics-exporter"
        }
      }

      spec {
        # Required to access libtpu monitoring APIs on the host
        host_network = true
        dns_policy   = "ClusterFirstWithHostNet"

        # Tolerate TPU taint so the pod can be scheduled on TPU nodes
        toleration {
          key      = "google.com/tpu"
          operator = "Exists"
          effect   = "NoSchedule"
        }

        # Run on any node that has a TPU accelerator, regardless of type or topology
        affinity {
          node_affinity {
            required_during_scheduling_ignored_during_execution {
              node_selector_term {
                match_expressions {
                  key      = "cloud.google.com/gke-tpu-accelerator"
                  operator = "Exists"
                }
              }
            }
          }
        }

        container {
          name  = "metrics-exporter"
          image = var.metrics_exporter_image

          port {
            name           = "http"
            container_port = 9000
            protocol       = "TCP"
          }

          resources {
            requests = {
              cpu    = "100m"
              memory = "256Mi"
            }
            limits = {
              cpu    = "500m"
              memory = "512Mi"
            }
          }
        }
      }
    }
  }

  depends_on = [google_container_cluster.primary]
}
