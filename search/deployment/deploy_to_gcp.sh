# Enable required services for project
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Create artifact registry repository
gcloud artifacts repositories create soundtrack-search \
    --repository-format=docker \
    --location=europe-west1 \
    --description="Docker repository for soundtrack-search"

# Auth docker with gcp
gcloud auth configure-docker europe-west1-docker.pkg.dev

# Build the image
docker build -t europe-west1-docker.pkg.dev/soundtrack-test-477015/soundtrack-search/search-app:latest .

# Push to Artifact Registry
docker push europe-west1-docker.pkg.dev/soundtrack-test-477015/soundtrack-search/search-app:latest

# Deploy to Cloud Run
gcloud run deploy search-app \
    --image europe-west1-docker.pkg.dev/soundtrack-test-477015/soundtrack-search/search-app:latest \
    --platform managed \
    --region europe-west1 \
    --allow-unauthenticated \
    --port 8000 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --min-instances 0 \
    --max-instances 10 

# Delete the service
gcloud run services delete search-app \
    --region europe-west1 \
    --quiet