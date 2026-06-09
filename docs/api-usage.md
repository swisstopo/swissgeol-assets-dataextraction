# API Usage Guide

## API Endpoints

The API supports two endpoint versions: **V1** (current) and **V2** (development).

**V1 Endpoints:**
- `/v1` — main document classification endpoint
- `/v1/collect` — response collection

**V2 Endpoints:**
- `/v2` — main document classification endpoint
- `/v2/collect` — response collection

The request JSON body structure for all endpoints follows the same pattern:

```json
{"file": "filename.pdf"}
```

## Page Classes

Each page is categorised into one of the following classes:

| Class | Description |
|-------|-------------|
| `Text` | Continuous text page |
| `Boreprofile` | Boreholes |
| `Maps` | Geological or topographic maps |
| `TitlePage` | Title pages of original reports |
| `GeoProfile` | Geological cross-sections or longitudinal profiles |
| `Table` | Tabular numeric/textual data |
| `Diagram` | Scientific 2D graphs or plots |
| `Unknown` | Everything else |

## Output Format

The output of the pipeline depends on the version queried.

### V1 Output

```jsonc
{
	"has_finished": true,
	"data": [
		{
			"filename": "input.pdf",					// Name of the file
			"metadata": {
				"page_count": 1,						// Number of pages
				"languages": [							// Detected languages
					"de"
				]
			},
			"pages": [									// List of dictionaries containing:
				{
					"predicted_class": "Boreprofile",	// Type of element (PascalPageClasses)
					"page_number": 1,					// The page number (1-indexed)
					"page_metadata": {
						"language": "de",				// Language of page
						"is_frontpage": false
					}
				}
			]
		}
	]
}
```

### V2 Output

```jsonc
{
	"has_finished": true,
	"data": {
		"filename": "input.pdf",					// Name of the file
		"page_count": 3,							// Number of pages
		"languages": [								// Detected languages
			"de"
		],
		"entities": [								// List of elements present in file
			{
				"classification": "boreprofile",	// Type of element (PageClasses)
				"language": "de",					// Detected language
				"page_start": 1,					// Starting page
				"page_end": 3,						// Ending page
				"title": "BS1"						// Entity title (None if not found)
			}
		]
	}
}
```

### General Notes

- The classifier supports batch input of multiple reports.
- Input must be preprocessed: PDFs should already have OCR.
- Classification is multi-class with a single label per page. Future updates may support multiple labels.

## Running the API Locally

Start the API server:

```bash
uvicorn api.api:app --reload --host 0.0.0.0 --port 8000
```

This starts the server on port 8000 with automatic reloading on code changes.

## MinIO Setup (Local S3)

To run the API on local documents instead of AWS S3, enable local S3 mode and spin up MinIO.

In your `.env` file, activate the local mode flag:

```bash
# Use local S3 (MinIO) instead of AWS
USE_LOCAL=True

# Bucket and prefix used by the API
S3_BUCKET="my-bucket"			# choose your own
S3_FOLDER="my-folder/"			# choose your own

# Local MinIO connection
LOCAL_S3_ENDPOINT="http://localhost:9000"
LOCAL_S3_ACCESS_KEY="admin"     # choose your own
LOCAL_S3_SECRET_KEY="admin123"  # choose your own
```

Replace `${LOCAL_S3_ACCESS_KEY}` / `${LOCAL_S3_SECRET_KEY}` with the values set in `.env`.

```bash
docker run -d --name minio \
  -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=${LOCAL_S3_ACCESS_KEY} \
  -e MINIO_ROOT_PASSWORD=${LOCAL_S3_SECRET_KEY} \
  -v "$(pwd)/minio/data:/data" \
  quay.io/minio/minio server /data --console-address ":9001"
```

Open the [MinIO UI](http://localhost:9001) console and log in using the credentials defined in your `.env` file. From the web interface, create a bucket named `${S3_BUCKET}`, then create a folder inside it called `${S3_FOLDER}`. Finally, upload your local PDF files to this folder. These files will then be available for the API when you run classification requests locally.

## Further Reading

For details on API versioning and shared components, see [api/README.md](../api/README.md).
