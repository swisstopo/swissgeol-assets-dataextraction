import logging
import sys
from pathlib import Path

from fastapi import FastAPI

from api.app.v1.router import router as v1_router
from api.app.v2.router import router as v2_router

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

app = FastAPI()

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

app.include_router(v1_router)
app.include_router(v2_router)
