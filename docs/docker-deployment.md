# Docker Deployment

## Building the Docker Image

Build the image:

```bash
docker build -t assets-api . -f Dockerfile
```

Verify the build:

```bash
docker images
```

## Running the Container

Run the Docker container, making sure your AWS credentials are configured in the `.env` file:

```bash
docker run -p 8000:8000 -v $(pwd)/.env:/app/.env.api:ro assets-api
```

## Testing the API

Run a classification job on a PDF file. Make sure the file exists in the configured S3 bucket and folder before starting.

```bash
# Run classification
curl -X POST http://127.0.0.1:8000/v1/ \
  -H "Content-Type: application/json" \
  -d '{"file": "YOUR_OWN_FILENAME.pdf"}' -i

# Collect results
curl -X POST http://127.0.0.1:8000/v1/collect \
  -H "Content-Type: application/json" \
  -d '{"file": "YOUR_OWN_FILENAME.pdf"}' -i
```

## Docker Compose

The repository includes a `docker-compose.yml` for convenience:

```bash
docker compose up
```

## MinIO

If you are using a local MinIO instance for file serving, `localhost` inside the container refers to the container itself, not your host machine. Update the endpoint in `.env` so the container can reach MinIO running on your host:

```bash
LOCAL_S3_ENDPOINT="http://host.docker.internal:9000"
```

## Run from Github Packages

The built Docker image for development is present on the repository under the Pakages. To run it locally use:

```
docker pull ghcr.io/swisstopo/swissgeol-assets-dataextraction-api:edge
docker run -p 8000:8000 -v $(pwd)/.env:/app/.env.api:ro ghcr.io/swisstopo/swissgeol-assets-dataextraction-api:edge
```