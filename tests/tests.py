# Tests for the taxi ride duration prediction API
#
# Run from the project root with:
#   uv run python -m pytest tests/tests.py -v

import importlib
import sys
from urllib.request import urlopen

import mlflow
import mlflow.pyfunc 
import pytest
from fastapi.testclient import TestClient

from app.helper_functions import FEATURE_COLUMNS, transform_request_input
from app.schemas import TaxiRideRequest
from src.config import MLFLOW_TRACKING_URI

VALID_RIDE = {
    "tpep_pickup_datetime": "2025-01-15T08:30:00",  # Wednesday
    "PULocationID": 161,
    "DOLocationID": 236,
    "trip_distance": 3.5,
}


class FakeModel:
    """Stand-in for the MLflow model: predicts 10 minutes per mile and remembers its input."""

    def __init__(self):
        self.last_input = None

    def predict(self, df):
        self.last_input = df
        return (df["trip_distance"] * 10).to_numpy()


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def client(fake_model):
    """Import the app with MLflow's model loading replaced by the fake model."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mlflow.pyfunc, "load_model", lambda uri: fake_model)
        # app/app.py loads the model at import time, so re-import it while the patch is active
        sys.modules.pop("app.app", None)
        app_module = importlib.import_module("app.app")
        yield TestClient(app_module.app, raise_server_exceptions=False)
    sys.modules.pop("app.app", None)


# --- Health check ---

def test_health_endpoint_returns_message(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "NYC Taxi Ride Duration Prediction"}


# --- Successful predictions ---

def test_predict_list_returns_one_prediction_per_ride(client):
    rides = [VALID_RIDE, {**VALID_RIDE, "trip_distance": 1.0}]
    response = client.post("/predict", json=rides)
    assert response.status_code == 200
    body = response.json()
    assert len(body) == len(rides)
    for item in body:
        assert isinstance(item["predicted_duration"], float)


def test_predict_single_ride_returns_list_of_one(client):
    response = client.post("/predict", json=VALID_RIDE)
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_response_echoes_request_fields(client):
    response = client.post("/predict", json=[VALID_RIDE])
    item = response.json()[0]
    for field in ["PULocationID", "DOLocationID", "trip_distance"]:
        assert item[field] == VALID_RIDE[field]
    assert item["tpep_pickup_datetime"].startswith("2025-01-15T08:30:00")


def test_predictions_keep_request_order(client):
    distances = [5.0, 1.0, 3.0]
    rides = [{**VALID_RIDE, "trip_distance": d} for d in distances]
    body = client.post("/predict", json=rides).json()
    # FakeModel predicts 10 min per mile, so each output must match its own input
    assert [item["predicted_duration"] for item in body] == [50.0, 10.0, 30.0]


def test_model_receives_training_features(client, fake_model):
    rides = [
        VALID_RIDE,                                                   # Wed 08:30
        {**VALID_RIDE, "tpep_pickup_datetime": "2025-01-19T23:05:00"},  # Sun 23:05
    ]
    client.post("/predict", json=rides)
    df = fake_model.last_input
    assert list(df.columns) == FEATURE_COLUMNS
    assert df["start_hour"].tolist() == [8, 23]
    assert df["start_weekday"].tolist() == [3, 7]


# --- Invalid requests are rejected with 422 ---

@pytest.mark.parametrize(
    "change, bad_field",
    [
        ({"PULocationID": 999}, "PULocationID"),
        ({"DOLocationID": 0}, "DOLocationID"),
        ({"trip_distance": 0}, "trip_distance"),
        ({"trip_distance": 100}, "trip_distance"),
        ({"tpep_pickup_datetime": "yesterday"}, "tpep_pickup_datetime"),
        ({"PULocationID": "abc"}, "PULocationID"),
        ({"trip_distnce": 3.5}, "trip_distnce"),  # typo → extra field forbidden
    ],
)
def test_invalid_field_is_rejected(client, change, bad_field):
    ride = {**VALID_RIDE, **change}
    response = client.post("/predict", json=[ride])
    assert response.status_code == 422
    error_locations = [err["loc"] for err in response.json()["detail"]]
    assert any(bad_field in loc for loc in error_locations)


def test_missing_field_is_rejected(client):
    ride = {k: v for k, v in VALID_RIDE.items() if k != "trip_distance"}
    response = client.post("/predict", json=[ride])
    assert response.status_code == 422


def test_missing_body_is_rejected(client):
    response = client.post("/predict")
    assert response.status_code == 422


def test_empty_list_is_rejected(client):
    response = client.post("/predict", json=[])
    assert response.status_code == 422


# --- Transform function on its own ---

def test_transform_output_dtypes_match_training():
    df = transform_request_input([TaxiRideRequest(**VALID_RIDE)])
    assert df.dtypes.astype(str).to_dict() == {
        "PULocationID": "int32",
        "DOLocationID": "int32",
        "trip_distance": "float64",
        "start_hour": "int8",
        "start_weekday": "int8",
    }


# --- Integration test with the real production model ---

def mlflow_is_running():
    try:
        urlopen(MLFLOW_TRACKING_URI, timeout=2)
        return True
    except OSError:
        return False


@pytest.mark.skipif(not mlflow_is_running(), reason="MLflow server is not running")
def test_real_model_gives_plausible_prediction():
    sys.modules.pop("app.app", None)
    app_module = importlib.import_module("app.app")
    client = TestClient(app_module.app)
    body = client.post("/predict", json=[VALID_RIDE]).json()
    # The model was trained on trips of 5–120 minutes
    assert 5 <= body[0]["predicted_duration"] <= 120
    sys.modules.pop("app.app", None)
