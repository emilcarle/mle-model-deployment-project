# Function to process the raw data into a preprocessed format
from pathlib import Path
import polars as pl

def transform_training_data(input_folder: Path, output_folder: Path) -> pl.DataFrame:
    # Calculate total revenue asd revenue per trip average for each month and in total
    # Use scan_parquet and streaming to reduce memory usage
    transformed_data = pl.scan_parquet(
        str(input_folder / "*.parquet")
        ).select(
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
        "trip_distance"
        ).with_columns(
            # Create a start time column (start_time)
            pl.col("tpep_pickup_datetime").dt.hour().alias("start_hour"),
            # Create a column with the trip duration (trip_duration)
            (pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime")).dt.total_seconds().truediv(60).alias("trip_duration_min"),
            # Create a weekday column (start_weekday)
            pl.col("tpep_pickup_datetime").dt.weekday().alias("start_weekday"),
        ).filter(
            # Remove trip times and distance that are unrealistic
            pl.col("trip_duration_min").is_between(5, (60*2)),
            pl.col("trip_distance").is_between(0.1, 50)
        ).drop(
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
        ).collect(
            engine="streaming"
        )
    # Write the raw data to disk as processed data
    transformed_data.write_parquet(output_folder / "processed_data.parquet")
    return transformed_data