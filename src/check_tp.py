"""Quick sanity check on downloaded total_precipitation file."""
import xarray as xr
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "era5" / "era5_tp.nc"

if not DATA_PATH.exists():
    print(f"NOT FOUND: {DATA_PATH}")
    exit(1)

ds = xr.open_dataset(DATA_PATH)
tp_var = list(ds.data_vars)[0]
tp = ds[tp_var].values
n_zero = (tp < 1e-6).sum()
n_total = tp.size
print(f"shape={tp.shape}, mean={tp.mean():.3f}, max={tp.max():.2f}, "
      f"zeros={n_zero}/{n_total} ({100*n_zero/n_total:.1f}%)")

# Quick spot-check across years
times = ds.time.values
print(f"Time range: {times[0]} to {times[-1]}")
print(f"Total time steps: {len(times)}")
ds.close()
