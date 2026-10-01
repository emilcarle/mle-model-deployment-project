# schemas defining the format for input and output data for the API

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field

class TaxiRideRequest(BaseModel):
    """
    Raw fields a client sends to the API as JSON for one taxi ride.
    """
    # Reject unknown keys (e.g. typos) instead of silently ignoring them
    model_config = ConfigDict(extra="forbid")
    tpep_pickup_datetime: datetime = Field(
        description="Date and time the trip starts, in ISO 8601 format.",
        examples=["2025-01-15T08:30:00"],
    )
    PULocationID: int = Field(
        ge=1,
        le=265,
        description="NYC TLC taxi zone ID where the trip starts.",
        examples=[161],
    )
    DOLocationID: int = Field(
        ge=1,
        le=265,
        description="NYC TLC taxi zone ID where the trip ends.",
        examples=[236],
    )
    trip_distance: float = Field(
        ge=0.1,
        le=50,
        description="Trip distance in miles. Limited to the range the model was trained on.",
        examples=[3.5],
    )

class TaxiRide(BaseModel):
    """
    Fields the API needs to describe one taxi ride.
    """
    PULocationID: int = Field(
        gt=0,
        description="NYC taxi pickup zone identifier used to build the route feature.",
        examples=[161],
    )
    DOLocationID: int = Field(
        gt=0,
        description="NYC taxi dropoff zone identifier used to build the route feature.",
        examples=[236],
    )
    trip_distance: float = Field(
        gt=0,
        description="Trip distance in miles. The demo model treats this as a numeric feature.",
        examples=[3.5],
    )
    start_hour: int = Field(
        ge=0,
        le=23,
        description="Hour associated with the start of the trip",
        examples=[5],
    )
    start_weekday: int = Field(
        description="The day of the week as an integer for the start of the trip",
        examples=[2],
    )

class TaxiRidePrediction(TaxiRideRequest):
    """
    Original ride payload plus the model's predicted duration.
    """
    predicted_duration: float = Field(
        description="Predicted ride duration in minutes.",
        examples=[12.4],
    )