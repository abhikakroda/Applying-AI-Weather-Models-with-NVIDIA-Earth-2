from cartopy.feature import NaturalEarthFeature

# from earth2studio.data import GFS
# from earth2studio.models.px.sfno import VARIABLES
# from earth2studio.utils.time import to_time_array
from windpowerlib.data import store_turbine_data_from_oedb

if __name__ == "__main__":
    _ = store_turbine_data_from_oedb()

    for resolution in ["10m", "50m", "110m"]:
        _ = NaturalEarthFeature("physical", "coastline", "10m").geometries()

    # gfs = GFS()
    # gfs(to_time_array(["2025-07-01"]), VARIABLES)
