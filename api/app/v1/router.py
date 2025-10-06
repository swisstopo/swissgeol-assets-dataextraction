import logging
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response, status
from starlette.responses import JSONResponse

from api.aws import aws
from api.utils import task
from api.utils.mapping import map_labels_for_app
from api.utils.schemas import CollectPayload, StartPayload
from api.utils.settings import ApiSettings, api_settings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import _apply_profile
from main import main as script

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

router = APIRouter(prefix="/v1")


@router.post("/")
def start(
    payload: StartPayload,
    settings: Annotated[ApiSettings, Depends(api_settings)],
    background_tasks: BackgroundTasks,
):
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


@router.post("/collect")
def collect(
    payload: CollectPayload,
):
    result = task.collect_result(payload.file)
    if result is None and not task.has_task(payload.file):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Page Classification is not running for this file"},
        )

    has_finished = result is not None
    if not has_finished:
        logging.info(f"Processing of '{payload.file}' has not yet finished.")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "has_finished": False,
                "data": None,
            },
        )

    if result.ok:
        logging.info(f"Processing of '{payload.file}' has been successful.")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "has_finished": True,
                "data": result.value,
            },
        )

    logging.info(f"Processing of '{payload.file}' has failed.")
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "has_finished": True,
            "error": "Internal Server Error",
        },
    )


def process(
    payload: StartPayload,
    aws_client: aws.Client,
    settings: Annotated[ApiSettings, Depends(api_settings)],
):
    task_id = f"{uuid.uuid4()}"
    tmp_dir = Path(settings.tmp_path) / task_id
    os.makedirs(tmp_dir, exist_ok=True)

    input_path = tmp_dir / "input.pdf"

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
    # the call to _apply_profile in script is based on environment variables, we make sure the results are mapped here.
    result = _apply_profile(result, "stable")

    result = [map_labels_for_app(doc) for doc in result]
    shutil.rmtree(tmp_dir)
    return result
