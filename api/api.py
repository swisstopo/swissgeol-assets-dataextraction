import logging
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Annotated

from aws import aws
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from utils import task
from utils.settings import ApiSettings, api_settings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import main as script

app = FastAPI()

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class StartPayload(BaseModel):
    """Payload for starting a new task."""

    file: str = Field(min_length=5)


@app.post("/")
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


class CollectPayload(BaseModel):
    """Payload for collecting task results."""

    file: str = Field(min_length=1)


@app.post("/collect")
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
        model_path="models/xgboost/stable/model.joblib",
        write_result=False,
    )
    shutil.rmtree(tmp_dir)
    return result
