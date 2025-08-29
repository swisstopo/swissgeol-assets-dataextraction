from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings


class ApiSettings(BaseSettings):
    """Settings for the API."""

    tmp_path: str

    aws_profile: str | None = None

    s3_bucket: str
    s3_folder: str

    use_local: bool = False
    local_s3_endpoint: str | None = None
    local_s3_access_key: str | None = None
    local_s3_secret_key: str | None = None


print("Loading env variables from '.env'.")
load_dotenv()

env_file = ".env.api"

print(f"Loading env variables from '{env_file}'.")
load_dotenv(env_file)


@lru_cache
def api_settings():
    return ApiSettings()
