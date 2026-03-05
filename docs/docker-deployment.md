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
