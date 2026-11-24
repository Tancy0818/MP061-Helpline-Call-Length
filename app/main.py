"""FastAPI service. Only the eight prediction-time inputs are accepted."""
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Literal
import uuid

from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    queue_wait_sec: float = Field(..., ge=0)
    agent_exp_months: float = Field(..., ge=0)
    previous_calls: int = Field(..., ge=0, le=20)
    issue_count: int = Field(..., ge=1)
    hour: int = Field(..., ge=0, le=23)
    caller_age: float = Field(..., ge=0)
    language: Literal["hindi", "english", "regional"]
    team: Literal["Team Alpha", "Team Beta", "Team Gamma"]


class PredictResponse(BaseModel):
    prediction: float


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {"timestamp": self.formatTime(record), "level": record.levelname,
                   "event": record.getMessage()}
        payload.update(getattr(record, "event_fields", {}))
        return json.dumps(payload, allow_nan=False)


def create_app(model_loader=None):
    logger = logging.getLogger("mp061.predictions")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.handlers = [handler]

    @asynccontextmanager
    async def lifespan(application):
        if model_loader is None:
            root = Path(__file__).resolve().parents[1]
            path = Path(os.getenv("MODEL_PATH", str(root / "models/model.joblib")))
            application.state.model = joblib.load(path)
        else:
            application.state.model = model_loader()
        yield

    application = FastAPI(title="MP061 Helpline Call Length", lifespan=lifespan)

    @application.get("/health")
    def health():
        return {"status": "ok"}

    @application.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest):
        started = time.perf_counter()
        request_id = str(uuid.uuid4())
        try:
            frame = pd.DataFrame([request.model_dump()])
            value = float(application.state.model.predict(frame)[0])
            if not np.isfinite(value):
                raise RuntimeError("Model returned a nonfinite prediction")
        except Exception:
            logger.error("prediction_failed", extra={"event_fields": {"request_id": request_id}})
            raise
        # Log operational metadata only, never the customer's request body.
        logger.info("prediction", extra={"event_fields": {
            "request_id": request_id, "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "status": "ok",
        }})
        return {"prediction": value}

    return application


app = create_app()
