import os
import time
from datetime import datetime, timedelta
from math import ceil, floor

import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm

LAT_LONS = {
    "Houston": (29.75, 264.75),
    "New Orleans": (30.0, 270.0),
    "San Francisco": (37.75, 237.5),
    "San Jose": (37.5, 238.0),
    "Tampa": (28.0, 277.5),
    "Paris": (48.75, 2.25),
    "London": (51.5, 0.0),
    "Munich": (48.0, 11.5),
    "Athens": (38.0, 23.75),
    "Cairo": (30.0, 31.25),
    "Nairobi": (-1.25, 36.75),
    "Cape Town": (-34.0, 18.5),
    "Caracas": (10.5, 293.0),
    "Rio de Janeiro": (-23.0, 316.75),
    "Lima": (-12.0, 283.0),
    "Bangkok": (13.75, 100.5),
    "Taipei": (25.0, 121.5),
    "Tokyo": (35.75, 139.75),
    "Manila": (14.5, 121.0),
    "Vientiane": (18.0, 102.5),
    "Melbourne": (-37.75, 145.0),
    "Wellington": (-41.25, 174.75),
    "Suva": (-18.25, 178.5),
}

TCS = {
    # name: (recommended start, recommended location)
    "Harvey": ("2017-08-24 12:00:00", "Houston"),
    "Ida": ("2021-08-28 12:00:00", "New Orleans"),
    "Chanthu": ("2021-09-12 00:00:00", "Taipei"),
    "Ian": ("2022-09-26 12:00:00", "Tampa"),
    "Noru": ("2022-09-26 12:00:00", "Vientiane"),
    "Beryl": ("2024-07-04 00:00:00", "Houston"),
    "Geemi": ("2024-07-23 00:00:00", "Taipei"),
    "Helene": ("2024-09-25 00:00:00", "Tampa"),
    "Kong-Rey": ("2024-10-30 00:00:00", "Taipei"),
}


def get_locations() -> list[str]:
    return list(LAT_LONS)


def get_lat_lon(loc: str) -> tuple[float, float]:
    if loc not in LAT_LONS:
        logger.error("Could not find '%s', returning Houston coordinates", loc)
        loc = "Houston"
    return LAT_LONS[loc]


def check_cds() -> None:
    cds_api = os.path.join(os.path.expanduser("~"), ".cdsapirc")
    if not os.path.exists(cds_api):
        key = input(
            "Enter CDS access token (e.g., 12345678-1234-1234-1234-123456123456):"
        )
        with open(cds_api, "w") as f:
            f.write("url: https://cds.climate.copernicus.eu/api\n")
            f.write(f"key: {key}\n")


def get_recent_time() -> np.datetime64:
    tmp = (datetime.now() - timedelta(hours=6)).replace(
        minute=0, second=0, microsecond=0
    )
    return np.datetime64(tmp.replace(hour=(tmp.hour // 6) * 6).isoformat())


@torch.no_grad()
def specific_to_relative(
    q: torch.Tensor, p: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    # See also
    # https://nvidia.github.io/earth2studio/examples/extend/03_custom_datasource.html
    epsilon = 0.621981

    e = (p * q * (1.0 / epsilon)) / (1 + q * (1.0 / (epsilon) - 1))

    es_w = 611.21 * torch.exp(17.502 * (T - 273.16) / (T - 32.19))
    es_i = 611.21 * torch.exp(22.587 * (T - 273.16) / (T + 0.7))

    alpha = torch.clip((T - 250.16) / (273.16 - 250.16), 0, 1.2) ** 2
    es = alpha * es_w + (1 - alpha) * es_i
    return 100 * e / es


def align_coords(*coords, factor: int = 4) -> list[float]:
    return [round(c * factor) / factor for c in coords]


def make_quarter_degree(coord_from: float, coord_to: float) -> np.ndarray:
    start = floor(coord_from * 4) / 4
    stop = ceil(coord_to * 4) / 4
    num = int(1 + 4 * (stop - start))
    return np.linspace(start, stop, num, endpoint=True)


def get_size(target):
    """
    Compute the size of a file, a folder (recursively), or a list of files/folders.

    Parameters:
      target: A single path (string) or a list of paths.

    Returns:
      Total size in bytes.
    """
    if isinstance(target, list):
        return sum(get_size(t) for t in target)
    if os.path.isdir(target):
        total = 0
        for root, _, files in os.walk(target):
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    total += os.path.getsize(filepath)
                except OSError:
                    pass  # skip files that can't be accessed
        return total
    elif os.path.isfile(target):
        return os.path.getsize(target)
    return 0


def monitor_progress(
    target,
    expected_total_size,
    *,
    baseline_size=0,
    desc="Processing",
    polling_interval=1,
):
    """
    Monitor the progress of a file or folder (or a list of them) until the size reaches the expected value.

    Parameters:
      target: A single path or list of paths to monitor.
      expected_total_size: The expected total size (in bytes) after the operation completes.
      baseline_size: Optional. The size (in bytes) already present before the operation.
      desc: A description string for the progress bar.
      polling_interval: Time in seconds between size checks.
    """
    # Calculate how much new data is expected.
    expected_progress = expected_total_size - baseline_size
    current_progress = get_size(target)

    if current_progress >= expected_progress:
        print(f"{desc} complete ({get_size(target)} bytes)")
        return

    with tqdm(total=expected_progress, desc=desc, unit="B", unit_scale=True) as pbar:
        previous_progress = current_progress
        pbar.update(previous_progress)
        while True:
            current_progress = get_size(target) - baseline_size
            delta = current_progress - previous_progress
            if delta > 0:
                pbar.update(delta)
                previous_progress = current_progress
            if current_progress >= expected_progress:
                # Make sure the progress bar is fully updated.
                if current_progress < expected_progress:
                    pbar.update(expected_progress - current_progress)
                print(f"{desc} complete ({get_size(target)} bytes)")
                break
            time.sleep(polling_interval)


def wait_for_cache():
    monitor_progress(
        os.environ["EARTH2STUDIO_CACHE"], 9642388063, desc="Waiting for cache"
    )
