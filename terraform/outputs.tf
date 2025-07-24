output "gcs_bucket_name" {
  description = "The name of the created GCS bucket."
  value       = google_storage_bucket.forts_bucket.name
}

output "artifact_registry_repository" {
  description = "The name of the created Artifact Registry repository."
  value       = google_artifact_registry_repository.forts_repo.name
}
