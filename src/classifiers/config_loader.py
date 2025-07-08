import yaml
from dotenv import load_dotenv
import os

load_dotenv()

def load_config(config_path="config.yml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r") as f:
        return f.read()

def get_aws_config() -> dict:
    return {
        "region": os.environ.get("AWS_MODEL_REGION"),
        "model_id": os.environ.get("AWS_MODEL_ID"),
    }
