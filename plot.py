from datetime import datetime, timedelta
from typing import Optional

import cartopy.crs as ccrs
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import xarray as xr
from cartopy.io.ogc_clients import WMTSRasterSource
from earth2studio.perturbation import Brown, SphericalGaussian
from matplotlib import gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import align_coords

plt.rcParams["animation.html"] = "jshtml"

sns.set_style("whitegrid")
sns.set_palette("colorblind")


def figure_global(
    da: xr.DataArray,
    loc_lat: Optional[float] = None,
    loc_lon: Optional[float] = None,
    norm: Optional[mcolors.Normalize] = None,
    path: Optional[str] = None,
    cmap: str = "viridis",
    projection: Optional[str] = None,
    colorbar: bool = False,
):
    kwargs = {"cmap": cmap}
    if norm is not None:
        kwargs["norm"] = norm

    if projection == "Robinson":
        fig, ax = plt.subplots(
            subplot_kw={"projection": ccrs.Robinson()}, figsize=(8, 5)
        )
    else:
        proj = ccrs.NearsidePerspective(
            central_longitude=loc_lon, central_latitude=loc_lat
        )
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": proj})

    pc = ax.pcolormesh(da.lon, da.lat, da, transform=ccrs.PlateCarree(), **kwargs)

    ax.coastlines(linewidth=1, color="#FFFFFFCC")
    ax.gridlines(linestyle="--", color="#FFFFFF55")

    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    if colorbar:
        cax = ax.inset_axes([0.04, 0.02, 0.015, 0.4])
        cb = fig.colorbar(pc, cax=cax)
        cb.outline.set_edgecolor("black")

    fig.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=300, transparent=True)
    else:
        plt.show()


def plot_perturbations(
    values: list[float],
    lats: int = 121,
    lons: int = 240,
    loc_lat: float = None,
    loc_lon: float = None,
):
    base = torch.zeros((1, lats, lons))
    coords = {
        "batch": np.empty(0),
        "lat": np.linspace(90.0, -90.0, lats),
        "lon": np.linspace(0, 360, lons, endpoint=False),
    }

    if loc_lat is not None and loc_lon is not None:
        proj = ccrs.NearsidePerspective(
            central_longitude=loc_lon, central_latitude=loc_lat
        )
    else:
        proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(
        nrows=2, ncols=len(values), figsize=(12, 4), subplot_kw={"projection": proj}
    )

    for i in (0, 1):
        for val, axi in zip(values, ax[i]):
            if i == 0:
                axi.set_title(f"alpha={val}")
                pdata, _ = SphericalGaussian(alpha=val)(base, coords)
            else:
                axi.set_title(f"reddening={val}")
                pdata, _ = Brown(reddening=val)(base, coords)
            axi.pcolormesh(
                coords["lon"],
                coords["lat"],
                pdata.squeeze(),
                transform=ccrs.PlateCarree(),
                cmap="viridis",
            )
            axi.coastlines(linewidth=0.5)

    ax[0, 0].set_yticks([])
    ax[0, 0].set_ylabel("SphericalGaussian")
    ax[1, 0].set_yticks([])
    ax[1, 0].set_ylabel("Brown")

    fig.tight_layout()
    plt.show()


def animate_global(
    da: xr.DataArray,
    start_time: np.datetime64,
    max_frames: int = 32,
    pcolormesh_kwargs: Optional[dict] = None,
):
    print("Creating animation...", end="")
    if pcolormesh_kwargs is None:
        pcolormesh_kwargs = {}

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()}, figsize=(8, 5))

    im = ax.pcolormesh(
        da.lon,
        da.lat,
        da.isel(lead_time=0).squeeze(),
        transform=ccrs.PlateCarree(),
        **{"cmap": "coolwarm", **pcolormesh_kwargs},
    )

    ax.set_title(f"{start_time} (lead time: 0 h)")
    ax.coastlines(linewidth=1.0)

    def animate(step: int):
        print(".", end="")
        im.set_array(da.isel(lead_time=step).squeeze())
        step_time = np.datetime_as_string(start_time + da.lead_time[step], unit="s")
        ax.set_title(f"{step_time.replace('T', ' ')} (lead time: {6 * step} h)")

    fig.tight_layout()
    plt.close("all")

    return animation.FuncAnimation(
        fig,
        animate,
        min(max_frames, len(da.lead_time)),
        blit=False,
        repeat=False,
        interval=500,
    )


def animate_corrdiff_us(
    ds: xr.Dataset,
    var: str,
    start_time: np.datetime64,
    max_frames: int = 32,
    pcolormesh_kwargs: Optional[dict] = None,
):
    print("Creating animation...", end="")
    if pcolormesh_kwargs is None:
        pcolormesh_kwargs = {}

    proj = ccrs.LambertConformal(
        central_longitude=262.5, central_latitude=38.5, standard_parallels=(38.5, 38.5)
    )
    fig, ax = plt.subplots(subplot_kw={"projection": proj}, figsize=(8, 5))

    im = ax.pcolormesh(
        ds.lon,
        ds.lat,
        ds.isel(lead_time=0)[var].squeeze(),
        transform=ccrs.PlateCarree(),
        **{"cmap": "coolwarm", **pcolormesh_kwargs},
    )

    ax.set_title(f"{start_time} (lead time: 0 h)")
    ax.coastlines(linewidth=1.0)

    def animate(step: int):
        print(".", end="")
        im.set_array(ds.isel(lead_time=step)[var].squeeze())
        step_time = np.datetime_as_string(
            start_time + timedelta(hours=ds.lead_time[step]), unit="s"
        )
        ax.set_title(f"{step_time.replace('T', ' ')} (lead time: {6 * step} h)")

    fig.tight_layout()
    plt.close("all")

    return animation.FuncAnimation(
        fig,
        animate,
        min(max_frames, len(ds.lead_time)),
        blit=False,
        repeat=False,
        interval=500,
    )


def animate_local(
    da: xr.DataArray,
    start_time: np.datetime64,
    loc_lat: float,
    loc_lon: float,
    max_frames: int = 32,
    pcolormesh_kwargs: Optional[dict] = None,
    cb_label: Optional[str] = None,
):
    print("Creating animation...", end="")
    if pcolormesh_kwargs is None:
        pcolormesh_kwargs = {}

    da = da.sel(time=start_time).drop_vars("time")

    proj = ccrs.PlateCarree(central_longitude=loc_lon)
    fig, ax = plt.subplots(subplot_kw={"projection": proj}, figsize=(6, 5))

    loc_lat, loc_lon = align_coords(loc_lat, loc_lon)
    loc_da = da.sel(
        lon=slice(loc_lon - 12, loc_lon + 12), lat=slice(loc_lat + 12, loc_lat - 12)
    )
    im = ax.pcolormesh(
        loc_da.lon,
        loc_da.lat,
        loc_da.isel(lead_time=0),
        transform=ccrs.PlateCarree(),
        **{"cmap": "coolwarm", **pcolormesh_kwargs},
    )

    ax.set_title(f"{start_time} (lead time: 0 h)")
    ax.coastlines(linewidth=1.0)

    def animate(step: int):
        print(".", end="")
        im.set_array(loc_da.isel(lead_time=step))
        step_time = np.datetime_as_string(start_time + loc_da.lead_time[step], unit="s")
        ax.set_title(f"{step_time.replace('T', ' ')} (lead time: {6 * step} h)")

    # Inset colorbar
    vmin = pcolormesh_kwargs.get("vmin", loc_da.values.min())
    vmax = pcolormesh_kwargs.get("vmax", loc_da.values.max())
    norm = Normalize(vmin, vmax, clip=True)
    cmap = pcolormesh_kwargs.get("cmap", "coolwarm")

    cax = ax.inset_axes([0.04, 0.02, 0.015, 0.4])
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, label=cb_label)

    fig.tight_layout()
    plt.close("all")

    return animation.FuncAnimation(
        fig,
        animate,
        min(max_frames, len(loc_da.lead_time)),
        blit=False,
        repeat=False,
        interval=500,
    )


def plot_line_ensemble(
    da: xr.DataArray,
    start_time: np.datetime64,
    loc_lat: float,
    loc_lon: float,
    ylabel: str,
):
    loc_lat, loc_lon = align_coords(loc_lat, loc_lon)
    da = da.sel(time=start_time).drop_vars("time")
    loc_da = da.sel(lon=loc_lon, lat=loc_lat)
    lead_time = loc_da.lead_time / np.timedelta64(1, "h")

    if len(loc_da.ensemble) > 4:
        cmap = sns.color_palette("mako", as_cmap=True)
        palette = [cmap(x) for x in np.linspace(0.2, 0.7, len(loc_da.ensemble))]
    else:
        palette = None

    fig, ax = plt.subplots(nrows=3, figsize=(7, 6), sharex=True)

    mean = loc_da.mean("ensemble")
    palette = sns.color_palette(palette)
    for i in range(len(loc_da.ensemble)):
        ax[0].plot(lead_time, loc_da.isel(ensemble=i), color=palette[i])
        ax[1].plot(lead_time, loc_da.isel(ensemble=i) - mean, color=palette[i])

    ax[0].set_title("Ensemble members")
    ax[1].set_title("Deviation from ensemble mean")

    ax[2].plot(lead_time, loc_da.std("ensemble"), color=palette[0])
    ax[2].set_title("Standard deviation")
    ax[2].set_xlabel("Lead time [h]")

    for axi in ax:
        axi.set_ylabel(ylabel)
        axi.set_xlim(min(lead_time) - 2, max(lead_time) + 2)

    fig.tight_layout()
    plt.show()


def plot_global_ensemble(
    da: xr.DataArray,
    start_time: np.datetime64,
    loc_lat: float,
    loc_lon: float,
    max_samples: int = 3,
    cb_label: Optional[str] = None,
    pcolormesh_kwargs: Optional[dict] = None,
    ilead_time: int = -1,
):
    if pcolormesh_kwargs is None:
        pcolormesh_kwargs = {}
    max_samples = min(max_samples, len(da.ensemble))

    da = da.sel(time=start_time).drop_vars("time")

    proj = ccrs.NearsidePerspective(central_longitude=loc_lon, central_latitude=loc_lat)
    fig = plt.figure(figsize=((max_samples + 1.4) * 3.2, 3.5))
    # max_samples samples, colorbar, std, colorbar
    gs = gridspec.GridSpec(
        nrows=1, ncols=max_samples + 3, width_ratios=[1] * max_samples + [0.2, 1, 0.2]
    )
    ax = [plt.subplot(gs[i], projection=proj) for i in range(max_samples)]

    cmap = "mako"
    vmin = da.isel(lead_time=ilead_time).min()
    vmax = da.isel(lead_time=ilead_time).max()

    for i in range(max_samples):
        im = ax[i].pcolormesh(
            da.lon,
            da.lat,
            da.isel(lead_time=ilead_time, ensemble=i),
            transform=ccrs.PlateCarree(),
            **{"cmap": cmap, "vmin": vmin, "vmax": vmax, **pcolormesh_kwargs},
        )
        ax[i].coastlines(linewidth=1.0)
        ax[i].gridlines(linestyle="--")
        ax[i].set_title(f"Ensemble member #{i + 1}")
    cax = plt.subplot(gs[max_samples])
    cax.axis("off")
    cax = cax.inset_axes([0.00, 0.00, 0.2, 1.0])
    fig.colorbar(im, cax=cax, label=cb_label)

    sax = plt.subplot(gs[max_samples + 1], projection=proj)
    im = sax.pcolormesh(
        da.lon,
        da.lat,
        da.isel(lead_time=ilead_time).std(dim="ensemble"),
        transform=ccrs.PlateCarree(),
        cmap=pcolormesh_kwargs.get("cmap", cmap),
    )
    sax.coastlines(linewidth=1.0)
    sax.gridlines(linestyle="--")
    sax.set_title(f"Standard deviation ({len(da.ensemble)} members)")
    cax = plt.subplot(gs[max_samples + 2])
    cax.axis("off")
    cax = cax.inset_axes([0.00, 0.00, 0.2, 1.0])
    fig.colorbar(im, cax=cax, label=cb_label)

    fig.tight_layout()
    plt.show()


def plot_wind_power_output(
    ds,
    wt,
    start_time: np.datetime64,
):
    # Identify cut in and out wind speeds
    pc = wt.power_plant.power_curve
    cutin = pc[pc.value > 0].iloc[0].wind_speed
    cutout = pc[pc.value > 0].iloc[-1].wind_speed

    # Create extended power curve
    pc = pd.concat(
        [
            pd.DataFrame([(0, 0)], columns=pc.columns),
            pc,
            pd.DataFrame([(cutout, 0), (cutout + 1, 0)], columns=pc.columns),
        ]
    )

    ds = ds.sel(time=start_time).drop_vars("time")
    lead_time = ds.lead_time / np.timedelta64(1, "h")

    if len(ds.ensemble) > 4:
        cmap = sns.color_palette("mako", as_cmap=True)
        palette = [cmap(x) for x in np.linspace(0.2, 0.7, len(ds.ensemble))]
    else:
        palette = None

    # fig, ax = plt.subplots(ncols=3, figsize=(14, 5))
    fig = plt.figure(figsize=(14, 4))
    gs = gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[1, 0.6, 1])
    ax = [plt.subplot(gs[i]) for i in range(3)]

    palette = sns.color_palette(palette)
    for i in range(len(ds.ensemble)):
        ax[0].plot(
            lead_time,
            ds.ws_hh.isel(ensemble=i),
            color=palette[i],
            alpha=0.5,
            linestyle=":",
        )
        ax[2].plot(
            lead_time,
            ds.power_output.isel(ensemble=i),
            color=palette[i],
            alpha=0.5,
            linestyle=":",
        )

    ax[0].plot(lead_time, ds.ws_hh.mean(dim="ensemble"), color=palette[0])
    ax[2].plot(lead_time, ds.power_output.mean(dim="ensemble"), color=palette[0])

    ax[0].set_title("Wind speed at hub height")
    ax[0].set_xlabel("Lead time [h]")
    ax[0].set_ylabel("Wind speed [m/s]")
    ax[0].axhline(y=cutout, color="grey", linestyle="-")
    ax[0].axhline(y=cutin, color="grey", linestyle="-")

    ax[1].plot(pc.value / 1e6, pc.wind_speed, color=palette[0])
    ax[1].set_xlabel("Power output [MW]")
    ax[1].set_ylabel("Wind speed [m/s]")
    ax[1].set_title("Power curve of wind turbine")

    ax[2].set_title("Power output")
    ax[2].set_xlabel("Lead time [h]")
    ax[2].set_ylabel("Power output [MW]")

    ax[0].set_ylim(0, cutout + 1)
    ax[1].set_ylim(0, cutout + 1)

    fig.tight_layout()
    plt.show()


def plot_score_line(
    ds_score: xr.Dataset, start_time: np.datetime64, vars: list[str], ylabels: list[str]
):
    fig, ax = plt.subplots(nrows=len(vars), figsize=(6, 2 * len(vars)), sharex=True)

    def _plot_score(i: int, var: str, ylabel: str):
        ax[i].plot(
            ds_score.lead_time / np.timedelta64(1, "h"),
            ds_score[var].sel(time=start_time),
        )
        ax[i].set_title(var)
        ax[i].set_ylabel(ylabel)

    for i, (var, ylabel) in enumerate(zip(vars, ylabels)):
        _plot_score(i, var, ylabel)

    ax[-1].set_xlabel("Lead time [h]")

    fig.tight_layout()
    plt.show()


def plot_rank_examples():
    fig, ax = plt.subplots(ncols=5, figsize=(8, 1.7))

    xx = np.linspace(-1, 1, 20)
    ax[0].plot(xx, 1 + 0 * xx)
    ax[1].plot(xx, xx**2)
    ax[2].plot(xx, 1 - xx**2)
    ax[3].plot(xx, xx)
    ax[4].plot(xx, -xx)

    for axi in ax:
        axi.set_xticks([])
        axi.set_yticks([])

    ax[0].set_title("Perfect")
    ax[1].set_title("Underdispersive")
    ax[2].set_title("Overdispersive")
    ax[3].set_title("Negative bias")
    ax[4].set_title("Positive bias")

    fig.tight_layout()
    plt.show()


def plot_rank_histograms(
    rh: xr.Dataset, start_time: np.datetime64, vars: list[str], appr_n: int = 4
):
    lead_times = rh.lead_time[1 :: len(rh.lead_time) // appr_n]
    ncols = len(lead_times)

    fig, ax = plt.subplots(
        nrows=len(vars), ncols=ncols, figsize=(ncols * 2, 1.5 * len(vars))
    )

    def _plot_hist(i: int, var: str):
        for j, lead_time in enumerate(lead_times):
            tmp = rh.sel(time=start_time, lead_time=lead_time)[var]
            ax[i, j].plot(
                tmp.sel(histogram_data="bin_centers"),
                tmp.sel(histogram_data="bin_counts"),
            )
            ax[i, j].set_xticks([])
            ax[i, j].set_xlim(0, 1)
            ax[i, j].set_yticks([])
            ax[i, j].set_ylim(0, None)

            if i == 0:
                ax[i, j].set_title(f"{int(lead_time / np.timedelta64(1, 'h'))} h")

        ax[i, 0].set_ylabel(var)

    for i, var in enumerate(vars):
        _plot_hist(i, var)

    fig.tight_layout()
    plt.show()


def plot_downscaling(
    lo_res: xr.Dataset,
    hi_res: xr.Dataset,
    var: str,
    start_time: np.datetime64,
    max_times: int = 6,
    normalize: bool = False,
    cb_label: Optional[str] = None,
):
    lo_res = lo_res.sel(time=start_time).drop_vars("time")
    hi_res = hi_res.sel(time=start_time).drop_vars("time")
    max_times = min(max_times, len(hi_res.lead_time))

    proj = ccrs.PlateCarree(central_longitude=float(lo_res.lon[len(lo_res.lon) // 2]))
    fig, ax = plt.subplots(
        ncols=max_times,
        nrows=2,
        subplot_kw={"projection": proj},
        figsize=(max_times * 2.7, 5),
    )

    if normalize:
        vmin = min(lo_res[var].min(), hi_res[var].min())
        vmax = max(lo_res[var].max(), hi_res[var].max())
    else:
        vmin, vmax = None, None

    for i in range(max_times):
        im0 = ax[0, i].pcolormesh(
            lo_res.lon,
            lo_res.lat,
            lo_res[var].isel(lead_time=i),
            transform=ccrs.PlateCarree(),
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
        )
        im1 = ax[1, i].pcolormesh(
            hi_res.lon,
            hi_res.lat,
            hi_res[var].isel(lead_time=i, sample=0),
            transform=ccrs.PlateCarree(),
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
        )

        ax[0, i].coastlines(linewidth=1.0)
        ax[1, i].coastlines(linewidth=1.0)

        if i == 0:
            ax[0, i].set_title(
                f"{start_time.astype(datetime).strftime('%Y-%m-%d %H:%M')}"
            )
        else:
            ax[0, i].set_title(
                f"+{int(hi_res.lead_time[i] / np.timedelta64(1, 'h'))} h"
            )

    cax = make_axes_locatable(ax[0, -1]).append_axes(
        "right", size="5%", pad=0.1, axes_class=plt.Axes
    )
    fig.colorbar(im0, cax=cax, label=cb_label)
    cax = make_axes_locatable(ax[1, -1]).append_axes(
        "right", size="5%", pad=0.1, axes_class=plt.Axes
    )
    fig.colorbar(im1, cax=cax, label=cb_label)

    # Colorbar messes up sizes, make all axes the same size
    for axi in ax.flatten():
        axi.set_aspect("auto", adjustable=None)

    ax[0, 0].set_yticks([])
    ax[0, 0].set_ylabel("GFS")
    ax[1, 0].set_yticks([])
    ax[1, 0].set_ylabel("CorrDiff")

    fig.tight_layout()
    plt.show()


def plot_downscaled_forecast(
    hi_res: xr.Dataset,
    var: str,
    start_time: np.datetime64,
    loc_lon: float,
    max_times: int = 6,
    cb_label: Optional[str] = None,
    add_wmts: bool = False,
):
    hi_res = hi_res.sel(time=start_time).drop_vars("time")
    max_times = min(max_times, len(hi_res.lead_time))

    proj = ccrs.PlateCarree(central_longitude=loc_lon)
    fig, ax = plt.subplots(
        ncols=max_times,
        subplot_kw={"projection": proj},
        figsize=(max_times * 2.7, 2.7),
    )

    if add_wmts:
        cmap = plt.get_cmap("coolwarm")
        cmap = cmap(np.arange(cmap.N))
        cmap[:, -1] = np.clip(np.exp(1.2 * np.linspace(0, 1, len(cmap))) - 1, 0, 1)
        cmap = ListedColormap(cmap)

        print("Retrieving BlueMarble imagery from NASA/Goddard Space Flight Center")
        url = "http://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi"
        layer = "BlueMarble_NextGeneration"
        wmts_source = WMTSRasterSource(url, layer)
    else:
        cmap = "coolwarm"

    for i in range(max_times):
        im = ax[i].pcolormesh(
            hi_res.lon,
            hi_res.lat,
            hi_res[var].isel(lead_time=i, sample=0),
            transform=ccrs.PlateCarree(),
            cmap=cmap,
        )

        ax[i].coastlines(linewidth=1.0)

        if i == 0:
            ax[i].set_title(f"{start_time.astype(datetime).strftime('%Y-%m-%d %H:%M')}")
        else:
            ax[i].set_title(f"+{int(hi_res.lead_time[i] / np.timedelta64(1, 'h'))} h")

        if add_wmts:
            ax[i].add_raster(wmts_source)

    cax = make_axes_locatable(ax[-1]).append_axes(
        "right", size="5%", pad=0.1, axes_class=plt.Axes
    )
    fig.colorbar(im, cax=cax, label=cb_label)

    # Colorbar messes up sizes, make all axes the same size
    for axi in ax:
        axi.set_aspect("auto", adjustable=None)

    ax[0].set_yticks([])
    ax[0].set_ylabel("CorrDiff")

    fig.tight_layout()
    plt.show()


def plot_downscaled_samples(
    hi_res: xr.Dataset,
    var: str,
    start_time: np.datetime64,
    loc_lon: float,
    max_samples: int = 4,
    cb_label: Optional[str] = None,
):
    hi_res = hi_res.sel(time=start_time).drop_vars("time")
    max_samples = min(max_samples, len(hi_res.sample))

    proj = ccrs.PlateCarree(central_longitude=loc_lon)

    fig = plt.figure(figsize=((max_samples + 1.4) * 3, 2.7))
    # max_samples samples, colorbar, std, colorbar
    gs = gridspec.GridSpec(
        nrows=1, ncols=max_samples + 3, width_ratios=[1] * max_samples + [0.2, 1, 0.2]
    )
    ax = [plt.subplot(gs[i], projection=proj) for i in range(max_samples)]

    for i in range(max_samples):
        im = ax[i].pcolormesh(
            hi_res.lon,
            hi_res.lat,
            hi_res[var].isel(lead_time=0, sample=i),
            transform=ccrs.PlateCarree(),
            cmap="coolwarm",
        )
        ax[i].coastlines(linewidth=1.0)
        ax[i].set_title(f"Sample #{i + 1}")
    cax = plt.subplot(gs[max_samples])
    cax.axis("off")
    cax = cax.inset_axes([0.00, 0.00, 0.2, 1.0])
    fig.colorbar(im, cax=cax, label=cb_label)

    sax = plt.subplot(gs[max_samples + 1], projection=proj)
    im = sax.pcolormesh(
        hi_res.lon,
        hi_res.lat,
        hi_res[var].isel(lead_time=0).std(dim="sample"),
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
    )
    sax.coastlines(linewidth=1.0)
    sax.set_title(f"Standard deviation ({len(hi_res.sample)} samples)")
    cax = plt.subplot(gs[max_samples + 2])
    cax.axis("off")
    cax = cax.inset_axes([0.00, 0.00, 0.2, 1.0])
    fig.colorbar(im, cax=cax, label=cb_label)

    ax[0].set_yticks([])
    ax[0].set_ylabel("CorrDiff")

    fig.tight_layout()
    plt.show()


def plot_hrrr_mini(inp: np.ndarray, out: np.ndarray):
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3))

    ax[0].imshow(inp, cmap="viridis")
    ax[1].imshow(out, cmap="viridis")

    ax[0].set_title("Input")
    ax[1].set_title("Target")
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_yticks([])

    fig.tight_layout()
    plt.show()


def plot_pop(
    pop_lo: np.ndarray,
    msk_lo: np.ndarray,
    pop_hi: np.ndarray,
    msk_hi: np.ndarray,
    lons_gfs: np.ndarray,
    lats_gfs: np.ndarray,
    lons_cd: np.ndarray,
    lats_cd: np.ndarray,
    loc_lon: float,
):
    proj = ccrs.PlateCarree(central_longitude=loc_lon)

    fig, ax = plt.subplots(
        ncols=4, figsize=(4 * 2.7, 2.7), subplot_kw={"projection": proj}
    )

    kwargs = dict(transform=ccrs.PlateCarree(), cmap="coolwarm")
    ax[0].pcolormesh(lons_gfs, lats_gfs, pop_lo, **kwargs)
    ax[0].set_title("Population (low-res)")
    ax[1].pcolormesh(lons_gfs, lats_gfs, msk_lo, **kwargs)
    ax[1].set_title("Mask (low-res)")
    ax[2].pcolormesh(lons_cd, lats_cd, pop_hi, **kwargs)
    ax[2].set_title("Population (high-res)")
    ax[3].pcolormesh(lons_cd, lats_cd, msk_hi, **kwargs)
    ax[3].set_title("Mask (high-res)")

    for axi in ax:
        axi.coastlines(linewidth=1.0)

    fig.tight_layout()
    plt.show()


def plot_pop_t2m(
    pop_gfs: xr.DataArray,
    msk_gfs: xr.DataArray,
    pop_cd: xr.DataArray,
    msk_cd: xr.DataArray,
    ylabel: str,
):
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(
        pop_gfs.lead_time / np.timedelta64(1, "h"),
        pop_gfs.isel(time=0),
        label="GFS (population)",
    )
    ax.plot(
        msk_gfs.lead_time / np.timedelta64(1, "h"),
        msk_gfs.isel(time=0),
        label="GFS (mask)",
    )
    ax.plot(
        pop_cd.lead_time / np.timedelta64(1, "h"),
        pop_cd.isel(time=0).mean(dim="sample"),
        label="CorrDiff (population)",
    )
    ax.plot(
        msk_cd.lead_time / np.timedelta64(1, "h"),
        msk_cd.isel(time=0).mean(dim="sample"),
        label="CorrDiff (mask)",
    )

    ax.set_xlabel("Lead time [h]")
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower right")

    fig.tight_layout()
    plt.show()
