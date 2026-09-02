import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# -----------------------------
# Step 1: Explicit File Paths
# -----------------------------
base_dir = r"C:\Users\Lenovo\Downloads\PFZ"

chl_path = os.path.join(base_dir, "AQUA_MODIS.20260101_20260131.L3m.MO.CHL.chlor_a.4km.nc")
sst_path = os.path.join(base_dir, "AQUA_MODIS.20260101_20260131.L3m.MO.NSST.sst.4km.nc")
par_path = os.path.join(base_dir, "AQUA_MODIS.20260101_20260131.L3m.MO.PAR.par.4km.nc")
kd_path  = os.path.join(base_dir, "AQUA_MODIS.20260101_20260131.L3m.MO.KD.Kd_490.4km.nc")

print("🔄 Opening NetCDF files from local directory...")

# -----------------------------
# Step 2: Load Data (Explicitly forcing NetCDF4 engine)
# -----------------------------
chl_ds = xr.open_dataset(chl_path, engine="netcdf4")
sst_ds = xr.open_dataset(sst_path, engine="netcdf4")
par_ds = xr.open_dataset(par_path, engine="netcdf4")
kd_ds  = xr.open_dataset(kd_path, engine="netcdf4")

# -----------------------------
# Step 3: Subset Bay of Bengal Region
# -----------------------------
lat_mask = (chl_ds['lat'] >= 5) & (chl_ds['lat'] <= 25)
lon_mask = (chl_ds['lon'] >= 80) & (chl_ds['lon'] <= 100)

chl = chl_ds['chlor_a'].where(lat_mask & lon_mask, drop=True)
sst = sst_ds['sst'].where(lat_mask & lon_mask, drop=True)
par = par_ds['par'].where(lat_mask & lon_mask, drop=True)
kd  = kd_ds['Kd_490'].where(lat_mask & lon_mask, drop=True)

# -----------------------------
# Step 4: Land Mask (remove invalid/land pixels)
# -----------------------------
mask = (~np.isnan(chl)) & (~np.isnan(sst)) & (~np.isnan(par)) & (~np.isnan(kd))
chl = chl.where(mask)
sst = sst.where(mask)
par = par.where(mask)
kd  = kd.where(mask)

# -----------------------------
# Step 5: Scoring Functions
# -----------------------------
def normalize(arr, min_val, max_val):
    return np.clip((arr - min_val) / (max_val - min_val), 0, 1)

def bell_shaped_score(arr, opt_val, tol):
    return np.exp(-((arr - opt_val) ** 2) / (2 * tol ** 2))

# -----------------------------
# Step 6: Calculate Scores
# -----------------------------
chl_score = normalize(chl, 0.1, 5.0)
sst_score = bell_shaped_score(sst, opt_val=26, tol=2)
par_score = bell_shaped_score(par, opt_val=40, tol=10)
kd_score  = 1 - normalize(kd, 0.05, 0.5)

# -----------------------------
# Step 7: Weighted PFZ Index
# -----------------------------
w_chl = 0.4
w_sst = 0.3
w_par = 0.2
w_kd  = 0.1

pfz_index = (w_chl * chl_score +
             w_sst * sst_score +
             w_par * par_score +
             w_kd  * kd_score)

# -----------------------------
# Step 8: PFZ Classification
# -----------------------------
# 0 = Low (0.0–0.4), 1 = Moderate (0.4–0.6), 2 = High (0.6–1.0)
pfz_class = xr.full_like(pfz_index, np.nan)

pfz_class = xr.where(pfz_index < 0.4, 0, pfz_class)
pfz_class = xr.where((pfz_index >= 0.4) & (pfz_index < 0.6), 1, pfz_class)
pfz_class = xr.where(pfz_index >= 0.6, 2, pfz_class)

# -----------------------------
# Step 9: Plotting
# -----------------------------
print("🎨 Generating your map...")
lon2d, lat2d = np.meshgrid(chl['lon'], chl['lat'])
cmap = plt.get_cmap('viridis', 3)
bounds = [-0.5, 0.5, 1.5, 2.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(10, 6))
pfz_plot = plt.pcolormesh(lon2d, lat2d, pfz_class, cmap=cmap, norm=norm, shading='auto')
cbar = plt.colorbar(pfz_plot, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(['Low', 'Moderate', 'High'])

plt.title('Satellite-Derived Multi-Parameter Potential Fishing Zone(PFZ) Classification – Bay of Bengal')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.tight_layout()
plt.show()


