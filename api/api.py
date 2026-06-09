import logging
import sys
from pathlib import Path

from fastapi import FastAPI

from api.app.v1.app import app as v1
from api.app.v2.app import app as v2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

app = FastAPI()

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

app.mount("/v1", v1)
app.mount("/v2", v2)
