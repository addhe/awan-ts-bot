#!/bin/bash

# A script to automate the deployment of the AI Trading Bot to Google Cloud Run.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
# Prompt user for variables if they are not already set
: "${GCP_PROJECT_ID?Please set GCP_PROJECT_ID environment variable or enter it here: }"
read -p "Enter GCP Project ID: " -i "$GCP_PROJECT_ID" GCP_PROJECT_ID

: "${GCP_REGION?Please set GCP_REGION environment variable or enter it here: }"
read -p "Enter GCP Region (e.g., us-central1): " -i "$GCP_REGION" GCP_REGION

SERVICE_NAME="ai-trading-bot"
SERVICE_ACCOUNT_NAME="ai-trading-bot-sa"
REPO_NAME="trading-bot-repo"

# --- Main Logic ---
echo "--- Starting Deployment to GCP Project: $GCP_PROJECT_ID in region $GCP_REGION ---"

# 1. Set gcloud config
echo "1. Setting gcloud project and region..."
gcloud config set project "$GCP_PROJECT_ID"
gcloud config set run/region "$GCP_REGION"

# 2. Enable necessary APIs
echo "2. Enabling required Google Cloud APIs..."
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com \
  iam.googleapis.com

# 3. Create Artifact Registry repository if it doesn't exist
echo "3. Setting up Artifact Registry..."
if ! gcloud artifacts repositories describe "$REPO_NAME" --location="$GCP_REGION" &>/dev/null; then
  echo "Creating Artifact Registry repository '$REPO_NAME'..."
  gcloud artifacts repositories create "$REPO_NAME" \
    --repository-format=docker \
    --location="$GCP_REGION" \
    --description="Docker repository for the AI Trading Bot"
else
  echo "Artifact Registry repository '$REPO_NAME' already exists."
fi

# 4. Create Service Account if it doesn't exist
echo "4. Setting up Service Account..."
SA_EMAIL="${SERVICE_ACCOUNT_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
if ! gcloud iam service-accounts describe "$SA_EMAIL" &>/dev/null; then
  echo "Creating service account '$SERVICE_ACCOUNT_NAME'..."
  gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
    --display-name="AI Trading Bot Service Account"
else
  echo "Service account '$SERVICE_ACCOUNT_NAME' already exists."
fi

# 5. Grant necessary IAM roles to the Service Account
echo "5. Granting IAM roles to the service account..."
# Role for Vertex AI (to use Gemini)
gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/aiplatform.user"
# Role to allow Cloud Run to be invoked (if needed in the future)
gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/run.invoker"

echo "IAM roles granted successfully."

# 6. Build the Docker image
echo "6. Building the Docker image..."
IMAGE_TAG="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:latest"
docker build -t "$IMAGE_TAG" .

# 7. Push the Docker image to Artifact Registry
echo "7. Pushing the Docker image to Artifact Registry..."
# Ensure docker is configured to authenticate with gcloud
gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev"
docker push "$IMAGE_TAG"

# 8. Deploy to Cloud Run
echo "8. Deploying the service to Cloud Run..."
# Check for .env file
if [ ! -f .env ]; then
    echo "ERROR: .env file not found. Please create it from .env.example and fill in your secrets."
    exit 1
fi

gcloud run deploy "$SERVICE_NAME" \
  --image="$IMAGE_TAG" \
  --service-account="$SA_EMAIL" \
  --set-env-vars-from-file=".env" \
  --allow-unauthenticated \
  --platform=managed \
  --region="$GCP_REGION"

echo "--- Deployment Complete! ---"
echo "Service URL: $(gcloud run services describe $SERVICE_NAME --platform=managed --region=$GCP_REGION --format='value(status.url)')"
