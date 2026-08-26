# Reviewed exports

The application writes reviewer-approved JSON exports to this directory.
Export files are ignored by Git because they may contain submitted text.

Docker Compose mounts this directory into the container so exports persist
after the container stops. Do not commit real tickets, confidential documents,
credentials, or personal data.
