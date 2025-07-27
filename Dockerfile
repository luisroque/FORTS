FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements files and install dependencies
COPY requirements.txt requirements.txt
COPY requirements-dev.txt requirements-dev.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy the rest of the application code
COPY . .

# Make the run script executable
RUN chmod +x ./run_all_experiments.sh

# Set the entrypoint to a bash shell, allowing arguments to be passed to the script
ENTRYPOINT ["/bin/bash", "./run_all_experiments.sh"]
