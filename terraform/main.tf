terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 4.50.0"
    }
  }
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

resource "google_project_service" "artifactregistry" {
  service = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "aiplatform" {
  service = "aiplatform.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "storage" {
  service = "storage.googleapis.com"
  disable_on_destroy = false
}

resource "google_storage_bucket" "forts_bucket" {
  name          = var.gcs_bucket_name
  location      = var.gcp_region
  force_destroy = true
  uniform_bucket_level_access = true

  depends_on = [google_project_service.storage]
}

resource "google_artifact_registry_repository" "forts_repo" {
  provider      = google-beta
  location      = var.gcp_region
  repository_id = var.ar_repo_name
  description   = "Docker repository for FORTS experiments"
  format        = "DOCKER"

  depends_on = [google_project_service.artifactregistry]
}

resource "google_project_iam_member" "vertex_ai_secret_accessor" {
  project = var.gcp_project_id
  role    = "roles/secretmanager.secretAccessor"

  # This grants the role to the default Vertex AI Custom Code Service Agent
  # See: https://cloud.google.com/vertex-ai/docs/general/access-control#service-agents
  member  = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-aiplatform-cc.iam.gserviceaccount.com"

  depends_on = [google_project_service.aiplatform]
}

resource "google_service_account" "vertex_ai_runner" {
  account_id   = "vertex-ai-runner"
  display_name = "Service Account for Vertex AI Jobs"
}

resource "google_project_iam_member" "vertex_ai_runner_secret_accessor" {
  project = var.gcp_project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.vertex_ai_runner.email}"
}

resource "google_project_iam_member" "vertex_ai_runner_ai_platform_user" {
  project = var.gcp_project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.vertex_ai_runner.email}"
}

resource "google_storage_bucket_iam_member" "vertex_ai_runner_storage_admin" {
  bucket = google_storage_bucket.forts_bucket.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.vertex_ai_runner.email}"
}

resource "google_service_account_iam_member" "vertex_ai_runner_user" {
  service_account_id = google_service_account.vertex_ai_runner.name
  role               = "roles/iam.serviceAccountUser"

  # This allows the job-submitting identity to act as the runner service account.
  # Replace with your principal if it's different.
  member             = "serviceAccount:artifact-pusher@${var.gcp_project_id}.iam.gserviceaccount.com"
}

data "google_project" "project" {}
