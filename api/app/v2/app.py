import logging
import os
import tempfile
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Response, status

from api.app.shared.handlers import start_handler
from api.app.shared.schemas import CollectPayload, ErrorResponse, StartPayload
from api.app.shared.settings import ApiSettings, api_settings
from api.app.v2.schemas import CollectResponse
from api.aws import aws
from api.utils import task
from main import main as script
from src.constants import DEFAULT_TREEBASED_MODEL_PATH

app = FastAPI()


@app.post(
    "/",
    status_code=204,
    summary="Start Page Classification",
    description="Starts the page classification process for a file as an asynchronous background task.",
    responses={
        400: {"model": ErrorResponse, "description": "Bad request (must be a PDF file)"},
        422: {"model": ErrorResponse, "description": "File does not exist in S3"},
    },
)
def start(
    payload: StartPayload,
    settings: Annotated[ApiSettings, Depends(api_settings)],
    background_tasks: BackgroundTasks,
) -> Response:
    return start_handler(payload, settings, background_tasks, process)


@app.post(
    "/collect",
    summary="Collect Page Classification Results",
    description="""
        Collects the results of the Page Classification process for a given file. If the process is still running,
        it indicates that the results are not yet available.
    """,
    response_model=CollectResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Classification not running for this file"},
        500: {"model": ErrorResponse, "description": "Internal server error during processing"},
    },
)
def collect(payload: CollectPayload) -> CollectResponse:
    result = task.collect_result(payload.file)
    if result is None and not task.has_task(payload.file):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Page classification is not running for this file"},
        )

    if result is None:
        logging.info(f"Processing of '{payload.file}' has not yet finished.")
        return CollectResponse(has_finished=False, data=None)

    if result.ok:
        logging.info(f"Processing of '{payload.file}'was successful.")
        return CollectResponse.create_response(result.value[0])

    logging.error(f"Processing of '{payload.file}' has failed.")
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={"message": "Internal Server Error during processing"},
    )


def process(
    payload: StartPayload,
    aws_client: aws.Client,
    settings: Annotated[ApiSettings, Depends(api_settings)],
) -> list[dict]:
    """Process a PDF file from S3 for page classification.

    Downloads the file from S3, runs the classification model, and returns the predictions.
    Creates a temporary directory for processing and cleans it up afterward.

    Args:
        payload: The payload containing the PDF file name to download and process
        aws_client: AWS client instance for S3 operations
        settings: API settings including S3 bucket, folder, and temp path configurations

    Returns:
        list[dict]: List of raw predictions from the classification pipeline. Note that these are raw pipeline outputs.
            Class labels will be translated to their final form during response object construction.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a file in the temp folder
        input_path = os.path.join(temp_dir, payload.file)

        # Load file locally to be processed
        aws.load_file(
            aws_client.bucket(settings.s3_bucket),
            f"{settings.s3_folder}{payload.file}",
            input_path,
        )

        # Process file in local directory
        _, entities = script(
            input_path=temp_dir,
            classifier_name="treebased",
            model_path=DEFAULT_TREEBASED_MODEL_PATH,
            write_result=False,
            return_entities=True,
        )

    # Return parsed entities
    return [entity.model_dump() for entity in entities]
