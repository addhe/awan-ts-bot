# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies that might be required by Python packages
# e.g., build-essential for compiling some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir makes the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Command to run the application
# Note: The main.py script has an infinite loop.
# In a containerized environment, it's often better to let the script exit
# and have the orchestrator (like Docker Compose or Kubernetes) handle restarts.
# This will be refactored later.
CMD [ "python", "app/main.py" ]
