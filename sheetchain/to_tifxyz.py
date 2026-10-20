# to_tifxyz.py GRIDS.npz NAME OUT_DIR : write one grid of a .npz (a merged sheet from merge_grids.py, or one fitted
# winding) as a tifxyz directory that VC3D and villa can open. The grid spacing in voxels is measured from the grid
# itself (median distance between neighbouring cells), so meta.json gets the matching scale.
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import tifxyz_out
src, name, out = sys.argv[1:4]
g = np.load(src)[name].astype(np.float32); ok = (g > 0).all(0)
d = np.linalg.norm(np.diff(g, axis=2), axis=0)[ok[:, 1:] & ok[:, :-1]]
step = float(np.median(d)); print(name, g.shape, "cell spacing", round(step, 2), "voxels")
tifxyz_out.write(os.path.join(out, f"{name}.tifxyz"), g, 1.0 / step, name=name)
