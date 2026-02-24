# Look up the default network and subnetwork for the cluster
data "google_compute_network" "default" {
  name    = "default"
  project = var.project_id
}

data "google_compute_subnetwork" "default" {
  name    = "default"
  region  = var.region
  project = var.project_id
}

# Create a private GKE cluster with a public endpoint.
# This is a common configuration enforced by organization policies.
resource "google_container_cluster" "primary" {
  name       = var.cluster_name
  location   = var.region
  project    = var.project_id
  network    = data.google_compute_network.default.self_link
  subnetwork = data.google_compute_subnetwork.default.self_link

  # Explicitly disable deletion protection for this development environment
  deletion_protection = false

  initial_node_count = 1

  release_channel {
    channel = "RAPID"
  }

  # Enable Workload Identity to allow Kubernetes service accounts to act as
  # Google Cloud service accounts. This provides a more secure way for workloads
  # to access Google Cloud APIs without using node service account credentials.
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  remove_default_node_pool = true

  # To use Managed Service for Prometheus, the legacy GKE monitoring must be disabled.
  monitoring_service = "none"

  # Enable Managed Service for Prometheus
  monitoring_config {
    managed_prometheus {
      enabled = true
    }
  }

}

resource "null_resource" "configure_kubectl" {
  provisioner "local-exec" {
    command = "gcloud container clusters get-credentials ${google_container_cluster.primary.name} --region ${var.region} --project ${var.project_id} && kubectl config set-context --current --namespace=default"
  }

  depends_on = [google_container_cluster.primary]
}

resource "google_container_node_pool" "v6e_2x4_node_pool" {
  project        = var.project_id
  cluster        = google_container_cluster.primary.name
  name           = "v6e-2x4-node-pool"
  location       = google_container_cluster.primary.location
  node_locations = var.node_locations
  node_count     = 2
  node_config {
    machine_type = "ct6e-standard-4t"
    spot         = true
  }

  placement_policy {
    type         = "COMPACT"
    tpu_topology = "2x4"
  }

  autoscaling {
    min_node_count = 0
    max_node_count = 2
  }
}

# CPU node pool for k8s system components
resource "google_container_node_pool" "system_node_pool" {
  project        = var.project_id
  cluster        = google_container_cluster.primary.name
  name           = "system-node-pool"
  location       = google_container_cluster.primary.location
  node_locations = var.node_locations
  node_count     = 1

  node_config {
    machine_type = "e2-highmem-8"

    # Enable GKE Metadata Server to support Workload Identity.
    # This allows pods to authenticate as Google Cloud service accounts
    # instead of using the node's service account.
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
}
