from collections.abc import Callable

from fastapi import BackgroundTasks, HTTPException
from starlette import status
from starlette.responses import Response

from api.app.shared.schemas import StartPayload
from api.app.shared.settings import ApiSettings
from api.aws import aws
from api.utils import task


def start_handler(
    payload: StartPayload,
    settings: ApiSettings,
    background_tasks: BackgroundTasks,
    process: Callable[[StartPayload, aws.Client, ApiSettings], list[dict] | None],
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
