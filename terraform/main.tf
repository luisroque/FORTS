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
