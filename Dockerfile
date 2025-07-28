FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements files and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Add build argument for GCP Project ID and set it as an environment variable
ARG FORTS_GCP_PROJECT_ID
ENV FORTS_GCP_PROJECT_ID=$FORTS_GCP_PROJECT_ID

# Copy the rest of the application code
COPY . .

# Make the run script executable
RUN chmod +x ./run_all_experiments.sh

# Set the entrypoint to a bash shell, allowing arguments to be passed to the script
ENTRYPOINT ["/bin/bash", "./run_all_experiments.sh"]
