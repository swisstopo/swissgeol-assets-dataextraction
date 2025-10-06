import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response, status

from api.aws import aws
from api.utils import task
from api.utils.schemas import (
    CollectPayload,
    CollectResponse,
    ErrorResponse,
    StartPayload,
)
from api.utils.settings import ApiSettings, api_settings
from main import main as script

router = APIRouter(prefix="/v2")


@router.post(
    "/",
    status_code=204,
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
    if not payload.file.endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"message": "input must be a PDF file"})

    aws_client = aws.connect(settings)
    has_file = aws_client.exists_file(
        settings.s3_bucket,
        f"{settings.s3_folder}{payload.file}",
    )
    if not has_file:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail={"message": "file does not exist"}
        )

    task.start(payload.file, background_tasks, lambda: process(payload, aws_client, settings))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/collect",
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
            detail={"message": "Page Classification is not running for this file"},
        )

    if result is None:
        logging.info(f"Processing of '{payload.file}' has not yet finished.")
        return CollectResponse(has_finished=False, data=None)

    if result.ok:
        logging.info(f"Processing of '{payload.file}' has been successful.")
        return CollectResponse.create_response(result.value)

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
        list[dict]: List of raw predictions from the classification pipeline. Note that these are raw pipeline outputs,
        class labels will be translated to their final form during response object construction.
    """
    task_id = f"{uuid.uuid4()}"
    tmp_dir = Path(settings.tmp_path) / task_id
    os.makedirs(tmp_dir, exist_ok=True)

    input_path = tmp_dir / payload.file

    aws.load_file(
        aws_client.bucket(settings.s3_bucket),
        f"{settings.s3_folder}{payload.file}",
        str(input_path),
    )

    result = script(
        input_path=tmp_dir,
        classifier_name="treebased",
        model_path="models/stable/model.joblib",
        write_result=False,
    )
    shutil.rmtree(tmp_dir)
    return result
