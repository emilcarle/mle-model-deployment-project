# File contains static variable definitions

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "taxi-duration-rf")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")

MLFLOW_EXPERIMENT_NAME = "taxi-trip-duration"

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/"
FILE_NAME_TEMPLATE = "yellow_tripdata_{month}.parquet"
MONTHS = ["2025-01", "2025-02"]
RAW_FOLDER = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_FOLDER = Path(__file__).resolve().parent.parent / "data" / "processed"