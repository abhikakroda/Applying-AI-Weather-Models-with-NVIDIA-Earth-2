import numpy as np
import xarray as xr
from windpowerlib import WindTurbine, modelchain
from windpowerlib.wind_speed import logarithmic_profile


def calculate_roughness_length(ds: xr.Dataset):
    rl_all = (10 ** (ds["s100m"] / (ds["s100m"] - ds["s10m"]))) * (
        100 ** (-ds["s10m"] / (ds["s100m"] - ds["s10m"]))
    )
    rl = np.nanmedian(rl_all)
    return rl


def calculate_power_output(ds: xr.Dataset, wt, loc_lat: float, loc_lon: float):
    assert loc_lon >= 0
    ds = ds.interp(lat=loc_lat, lon=loc_lon, method="linear")
    ds = ds.assign(s10m=np.sqrt(ds["u10m"] ** 2 + ds["v10m"] ** 2))
    ds = ds.assign(s100m=np.sqrt(ds["u100m"] ** 2 + ds["v100m"] ** 2))
    roughness_length = calculate_roughness_length(ds)
    ds = ds.assign(
        ws_hh=logarithmic_profile(
            ds["s100m"], 100, wt.power_plant.hub_height, roughness_length
        )
    )
    ds["power_output"] = (
        ["ensemble", "time", "lead_time"],
        wt.calculate_power_output(ds["ws_hh"], 0) / 1000000,
    )
    return ds


def get_wind_turbine_model(wind_turbine_specification):
    wt = WindTurbine(**wind_turbine_specification)
    modelchain_data = {"density_model": "ideal_gas"}
    wtmc = modelchain.ModelChain(wt, **modelchain_data)
    return wtmc
