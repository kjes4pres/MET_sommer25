import sys

sys.path.append("/home/kjsta7412/sommer_25/MET_sommer25")

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean.cm as cmo
from Rossby_deformation.funcs import *
from Rossby_deformation.get_turbine_coords import *
import matplotlib.style as mplstyle
import cartopy.feature as cfeature
from matplotlib.lines import Line2D

mplstyle.use(["ggplot", "fast"])

"""
Plots an overview map of the Norkyst v3 domain with all wind farms marked on the map.
"""

SV_F = get_turbine_coords(
    "/lustre/storeB/project/nwp/havvind/turbine_coordinates/windfarms_Sorvest_F.txt"
)
Gol = get_turbine_coords(
    "/lustre/storeB/project/nwp/havvind/turbine_coordinates/windfarms_Goliat.txt"
)
NV_CD = get_turbine_coords(
    "/lustre/storeB/project/nwp/havvind/turbine_coordinates/windfarms_Nordavind_CD.txt"
)
NV_C = get_turbine_coords(
    "/lustre/storeB/project/nwp/havvind/turbine_coordinates/windfarms_Nordvest_C.txt"
)
VV_B = get_turbine_coords(
    "/lustre/storeB/project/nwp/havvind/turbine_coordinates/windfarms_Vestavind_B.txt"
)
VV_F = get_turbine_coords(
    "/lustre/storeB/project/nwp/havvind/turbine_coordinates/windfarms_Vestavind_F.txt"
)

path = "/lustre/storeB/project/nwp/havvind/hav/results/experiment/EXP-03/norkyst_avg_0001.nc"
ds = xr.open_dataset(path)

fig, ax = plt.subplots(
    figsize=(6, 8), subplot_kw={"projection": ccrs.NorthPolarStereo()}
)

im = ax.pcolormesh(
    ds.lon_rho, ds.lat_rho, ds.h, transform=ccrs.PlateCarree(), cmap="cmo.deep"
)
cbar = plt.colorbar(im, shrink=0.8)
cbar.set_label("Bathymetry [m]")


# Define turbine colors and names for easier management
turbine_data = [
    (SV_F, "firebrick", "Sørvest-F"),
    (Gol, "darkgoldenrod", "Goliat"),
    (NV_CD, "olive", "Nordavind-CD"),
    (NV_C, "teal", "Nordvest-C"),
    (VV_B, "blueviolet", "Vestavind-B"),
    (VV_F, "hotpink", "Vestavind-F"),
]

for turbine_coords, color, label in turbine_data:
    ax.scatter(
        turbine_coords.coordinates[:, 0],
        turbine_coords.coordinates[:, 1],
        transform=ccrs.PlateCarree(),
        color=color,
        s=0.5,
        marker="*",
        label=label,
    )


# Adding land, coastline, and borders
land = cfeature.NaturalEarthFeature(
    category="physical",
    name="land",
    scale="50m",
    edgecolor="gray",
    facecolor=cfeature.COLORS["land"],
)
coastline = cfeature.NaturalEarthFeature(
    category="physical",
    name="coastline",
    scale="50m",
    edgecolor="gray",
    facecolor="none",
)
borders = cfeature.NaturalEarthFeature(
    category="cultural",
    name="admin_0_boundary_lines_land",
    edgecolor="gray",
    scale="50m",
    facecolor="none",
)

ax.add_feature(land)
ax.add_feature(coastline)
ax.add_feature(borders)

# Gridlines
gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=1,
    color="lightgray",
    alpha=0.5,
    linestyle="--",
)
gl.top_labels = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# Boundary lines
ax.plot(
    ds.lon_rho[0, :],
    ds.lat_rho[0, :],
    "--",
    transform=ccrs.PlateCarree(),
    color="gray",
    linewidth=0.8,
    label="Norkyst boundary",
)
ax.plot(
    ds.lon_rho[-1, :],
    ds.lat_rho[-1, :],
    "--",
    transform=ccrs.PlateCarree(),
    color="gray",
    linewidth=0.8,
)
ax.plot(
    ds.lon_rho[:, 0],
    ds.lat_rho[:, 0],
    "--",
    transform=ccrs.PlateCarree(),
    color="gray",
    linewidth=0.8,
)
ax.plot(
    ds.lon_rho[:, -1],
    ds.lat_rho[:, -1],
    "--",
    transform=ccrs.PlateCarree(),
    color="gray",
    linewidth=0.8,
)

# Create custom legend
legend_elements = [
    Line2D(
        [0], [0],
        marker='*',
        color='none',
        label=label,
        markerfacecolor=color,
        markersize=8, 
        markeredgewidth=0 
    )
    for _, color, label in turbine_data
]

legend_elements.insert(
    0,
    Line2D([0], [0], linestyle='--', color='gray', linewidth=0.8, label="Norkyst boundary")
)

ax.legend(handles=legend_elements, loc="upper left")

fig.tight_layout()
plt.savefig("Figures/overview_farms.png")
plt.show()
