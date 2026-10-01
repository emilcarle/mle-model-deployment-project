# Definition of the app serving predictions to the user


import mlflow
from pathlib import Path
from src.config import MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME, MODEL_ALIAS
from app.helper_functions import transform_request_input
from fastapi import FastAPI
from app.schemas import TaxiRidePrediction, TaxiRideRequest
from pydantic import Field
from typing import Annotated


PROJECT_ROOT = Path(__file__).resolve().parents[1]


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
model = mlflow.pyfunc.load_model(model_uri)


app = FastAPI(
    title="NYC Taxi Duration Prediction API",
    description=(
        "A FastAPI service that loads a registered MLflow model "
        "and returns taxi ride duration predictions."
    ),
    version="1.0.0",
)


@app.get("/", tags=["health"])
def index():
    """Return a simple message so we can verify the API is running."""
    return {"message": "NYC Taxi Ride Duration Prediction"}


@app.post("/predict", response_model=list[TaxiRidePrediction], tags=["predictions"])
def predict(request_taxi_rides: TaxiRideRequest | Annotated[list[TaxiRideRequest], Field(min_length=1)]):
    """
    Predict the trip duration in minutes for either a single or a set of taxi ride inputs.
    """
    if isinstance(request_taxi_rides, TaxiRideRequest):
        request_taxi_rides = [request_taxi_rides]
    df = transform_request_input(request_taxi_rides)
    predictions = model.predict(df)
    output = [
        TaxiRidePrediction(**record.model_dump(), predicted_duration=float(p))
        for record, p in zip(request_taxi_rides, predictions)
    ]
    return output