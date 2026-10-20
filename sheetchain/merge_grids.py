# merge_grids.py OUT.npz NAME Z0 Z1 UMB.json BAND1.npz w,w,... [BAND2.npz w,w,... ...]
# Join consecutive windings of each band into one multi-turn grid, resample every row onto a common unwrapped-angle
# axis (centre = umbilicus), keep rows with Z0 <= z < Z1 and stack the bands by z. Output key NAME, shape (3, rows, cols).
# A band may be given as BAND.npz:ZLO:ZHI to keep only its rows in [ZLO, ZHI) (overlapping band fits).
import sys, json, numpy as np
out, name, z0, z1 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
umb = json.load(open(sys.argv[5]))
bands = [(sys.argv[i], sys.argv[i + 1].split(",")) for i in range(6, len(sys.argv), 2)]
cp = sorted(umb["control_points"], key=lambda p: p["z"])
def centre(z):
    zs = [p["z"] for p in cp]; return np.interp(z, zs, [p["x"] for p in cp]), np.interp(z, zs, [p["y"] for p in cp])
rows = []
for bi, (f, ws) in enumerate(bands):
    f, *win = f.split(":"); blo, bhi = (float(win[0]), float(win[1])) if win else (-1e9, 1e9)
    G = np.load(f); g = np.concatenate([G[w].astype(np.float64) for w in ws], axis=2)
    for r in range(g.shape[1]):
        p = g[:, r]; m = (p > 0).all(0)
        if m.sum() < 0.8 * p.shape[1]:
            continue
        z = p[2, m].mean()
        if not (z0 <= z < z1 and blo <= z < bhi):
            continue
        cx, cy = centre(z); idx = np.where(m)[0]
        th = np.unwrap(np.arctan2(p[1, idx] - cy, p[0, idx] - cx))
        rows.append((z, th, p[:, idx], bi))
# common angle axis: overlap of all rows, same direction
sgn = np.sign(rows[0][1][-1] - rows[0][1][0])
for i, (z, th, p, bi) in enumerate(rows):          # make every row increasing in angle and start near the first row's start
    if np.sign(th[-1] - th[0]) != sgn:
        th, p = th[::-1], p[:, ::-1]
    th = th * sgn; k = np.round((rows[0][1][0] * sgn - th[0]) / (2 * np.pi)); rows[i] = (z, th + 2 * np.pi * k, p, bi)
lo = max(r[1][0] for r in rows); hi = min(r[1][-1] for r in rows)
n = int(round(np.median([len(r[1]) for r in rows]) * (hi - lo) / np.median([r[1][-1] - r[1][0] for r in rows])))
ax = np.linspace(lo, hi, n)
rows.sort(key=lambda r: r[0])
out_g = np.stack([np.stack([np.interp(ax, th, p[j]) for j in range(3)]) for z, th, p, bi in rows], axis=1).astype(np.float32)
# blend the jump at each band boundary: spread the xy offset over 4 rows on each side
zz = out_g[2].mean(1); src = np.array([r[3] for r in rows])
gaps = np.where((np.diff(zz) > 2.5 * np.median(np.diff(zz))) | (np.diff(src) != 0))[0]
for b in gaps:
    off = out_g[:2, b + 1] - out_g[:2, b]
    for k in range(4):
        w = 0.5 * (1 - k / 4)
        if b - k >= 0: out_g[:2, b - k] += w * off
        if b + 1 + k < out_g.shape[1]: out_g[:2, b + 1 + k] -= w * off
old = dict(np.load(out)) if __import__("os").path.exists(out) else {}
np.savez(out, **{**old, name: out_g})
zz = out_g[2].mean(1); print(name, out_g.shape, "z", round(zz.min()), round(zz.max()), "turns", round((hi - lo) / 2 / np.pi, 2),
      "max row gap", round(float(np.diff(zz).max()), 1))
