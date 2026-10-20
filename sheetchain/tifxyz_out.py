"""Write a grid surface as a tifxyz directory (x.tif, y.tif, z.tif, meta.json), the format VC3D and villa read.
Invalid cells are -1. `scale` is grid cells per voxel (a grid with 5 voxel spacing has scale 0.2)."""
import os, json, uuid as _uuid
import numpy as np
import tifffile


def write(path, grid, scale, name=None):
    """grid: (3, h, w) x/y/z in scan voxels, invalid <= 0. Returns the directory written."""
    g = np.asarray(grid, np.float32).copy(); ok = (g > 0).all(0); g[:, ~ok] = -1
    os.makedirs(path, exist_ok=True)
    for i, a in enumerate("xyz"):
        tifffile.imwrite(f"{path}/{a}.tif", g[i], compression="lzw", tile=(1024, 1024) if min(g.shape[1:]) >= 1024 else None)
    lo = [float(g[i][ok].min()) for i in range(3)]; hi = [float(g[i][ok].max()) for i in range(3)]
    json.dump({"uuid": name or str(_uuid.uuid4()), "scale": [float(scale), float(scale)], "type": "seg", "format": "tifxyz",
               "bbox": [lo, hi]}, open(f"{path}/meta.json", "w"), indent=2)
    return path
