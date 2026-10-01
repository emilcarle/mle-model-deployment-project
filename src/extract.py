# Functions to download the taxi dataset and save to disk

import requests
from pathlib import Path

def download_taxi_data(base_url: str, file_name_template: str, months: list, output_folder: Path):
    for month in months:
        # Create filename, url, and output path
        file_name = file_name_template.format(month=month)
        url = base_url + file_name
        output_filepath = output_folder / file_name
        # run the http request and check for status
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            # stream the files (incase they are large)
            with output_filepath.open("wb") as file:
                for chunk in response.iter_content(chunk_size=10000):
                    if chunk:
                        file.write(chunk)