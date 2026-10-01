# Functions for the app that handles prediction requests

import pandas as pd
import pandas as pd
import polars as pl
from pydantic import TypeAdapter
from app.schemas import TaxiRide, TaxiRideRequest, TaxiRidePrediction

FEATURE_COLUMNS = ["PULocationID", "DOLocationID", "trip_distance", "start_hour", "start_weekday"]
TAXIRIDEREQUEST_ADAPTER = TypeAdapter(list[TaxiRideRequest])
TAXIRIDE_ADAPTER = TypeAdapter(list[TaxiRide])

def transform_request_input(request_taxi_rides: list[TaxiRideRequest]) -> pd.DataFrame:
    """
    Turn one or more API ride dicts into a model-ready DataFrame.
    """
    df = pl.DataFrame([ride.model_dump() for ride in request_taxi_rides])
    df = df.with_columns(
        pl.col("PULocationID").cast(pl.Int32),
        pl.col("DOLocationID").cast(pl.Int32),
        pl.col("trip_distance").cast(pl.Float64),
        pl.col("tpep_pickup_datetime").dt.hour().alias("start_hour"),
        pl.col("tpep_pickup_datetime").dt.weekday().alias("start_weekday"),
    )
    df = df[FEATURE_COLUMNS]
    TAXIRIDE_ADAPTER.validate_python(df.to_dicts())
    return df.to_pandas()