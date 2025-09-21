import numpy as np
from earth2studio.data import GFS, NCAR_ERA5, WB2Climatology, fetch_data
from earth2studio.models.dx import CorrDiffTaiwan
from earth2studio.models.dx.corrdiff import VARIABLES as CORRDIFF_VARIABLES
from earth2studio.models.px import SFNO
from earth2studio.models.px.sfno import VARIABLES as SFNO_VARIABLES
from earth2studio.utils.time import to_time_array

if __name__ == "__main__":
    # Models
    _ = CorrDiffTaiwan.load_model(CorrDiffTaiwan.load_default_package())
    _ = SFNO.load_model(SFNO.load_default_package())

    gfs = GFS()
    ncar_era5 = NCAR_ERA5()
    wb2 = WB2Climatology()

    # Notebook 1
    _ = gfs(to_time_array(["2025-07-01"]), SFNO_VARIABLES)

    times = to_time_array(["2023-03-24 12:00:00", "2024-06-17 18:00:00"])
    variables = ["t2m", "tcwv", "u10m", "v10m"]
    _ = ncar_era5(times, variables)
    _ = gfs(times, variables)

    _ = ncar_era5(to_time_array(["2024-07-04 00:00:00"]), SFNO_VARIABLES)

    # Notebook 2
    start_time = np.datetime64("2024-07-04 00:00:00")
    nsteps = 32
    lead_time = np.array([np.timedelta64(6 * i, "h") for i in range(nsteps + 1)])
    _ = fetch_data(
        source=ncar_era5,
        time=[start_time],
        variable=["t2m", "u10m", "z500"],
        lead_time=lead_time,
    )
    _ = fetch_data(
        source=wb2,
        time=[start_time],
        variable=["t2m", "u10m", "z500"],
        lead_time=lead_time,
    )

    warmup_times = np.array(
        [start_time + np.timedelta64(6 * i, "h") for i in range(-3, 1)]
    )
    _ = fetch_data(
        source=ncar_era5,
        time=warmup_times,
        variable=SFNO_VARIABLES,
    )

    # Notebook 3
    start_time = np.datetime64("2024-07-01 12:00:00")
    gfs(to_time_array([start_time]), CORRDIFF_VARIABLES)

    nsteps = 12
    lead_time = np.array([np.timedelta64(6 * i, "h") for i in range(nsteps + 1)])
    _ = fetch_data(
        source=gfs,
        time=[start_time],
        variable=["t2m", "u10m", "v10m"],
        lead_time=lead_time,
    )
