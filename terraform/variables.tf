variable "gcp_project_id" {
  description = "The GCP project ID to use."
  type        = string
}

variable "gcp_region" {
  description = "The GCP region to use."
  type        = string
  default     = "us-central1"
}

variable "gcs_bucket_name" {
  description = "The name for the GCS bucket."
  type        = string
}

variable "ar_repo_name" {
  description = "The name for the Artifact Registry repository."
  type        = string
  default     = "forts-repo"
}
