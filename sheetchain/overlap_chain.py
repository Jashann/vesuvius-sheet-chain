# overlap_chain.py OUT.json x0,x1,y0,y1 BAND:grids.npz ... (bands in z order, neighbours overlapping in z)
# Chain fitted windings of overlapping band fits into sheets, inside one x/y sector. For each pair of neighbouring
# bands, every winding's surface in the shared z slab (upsampled 4x) is compared with every winding of the other band:
# D[a, b] = median distance from a's points to b's surface. A link a->b is kept when b is the best match both ways,
# the median distance is under MAXD voxels and the runner-up is clearly worse (ratio < RATIO). Without a z overlap
# the tool falls back to a +-60 voxel slab around the boundary (the case that was ambiguous before).
import sys, json, numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import zoom
MAXD, RATIO, MINPTS = 10.0, 0.6, 200
out = sys.argv[1]; x0, x1, y0, y1 = map(float, sys.argv[2].split(","))
bands = [a.split(":") for a in sys.argv[3:]]
G = {b: dict(np.load(f)) for b, f in bands}


def pts(a, z0, z1):                     # dense surface points of one winding inside the slab and the sector
    m = (a > 0).all(0).astype(np.float32)
    up = np.stack([zoom(a[j], 4, order=1) for j in range(3)]); mm = zoom(m, 4, order=1) > 0.99
    p = up[:, mm].T
    k = (p[:, 2] >= z0) & (p[:, 2] < z1) & (p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= y0) & (p[:, 1] < y1)
    return p[k]


def zrange(g):
    z = np.concatenate([a[2][(a > 0).all(0)] for a in g.values()]); return np.percentile(z, 1), np.percentile(z, 99)


links, report = {}, {}
for (b0, _), (b1, _) in zip(bands, bands[1:]):
    lo0, hi0 = zrange(G[b0]); lo1, hi1 = zrange(G[b1])
    za, zb = max(lo0, lo1) + 20, min(hi0, hi1) - 20
    if zb - za < 120:
        c = (za + zb) / 2; za, zb = c - 60, c + 60
    P0 = {w: pts(a, za, zb) for w, a in G[b0].items()}; P0 = {w: p for w, p in P0.items() if len(p) >= MINPTS}
    P1 = {w: pts(a, za, zb) for w, a in G[b1].items()}; P1 = {w: p for w, p in P1.items() if len(p) >= MINPTS}
    T0 = {w: cKDTree(p) for w, p in P0.items()}; T1 = {w: cKDTree(p) for w, p in P1.items()}
    w0s, w1s = sorted(P0), sorted(P1)
    D = np.full((len(w0s), len(w1s)), np.inf)
    for i, a in enumerate(w0s):
        c0 = P0[a].mean(0)
        for j, b in enumerate(w1s):
            if np.abs(P1[b].mean(0)[:2] - c0[:2]).max() > 400:
                continue
            d = T1[b].query(P0[a][::4])[0]; D[i, j] = np.median(d)
    L = {}
    for j, b in enumerate(w1s):
        col = D[:, j]; o = np.argsort(col)
        if not np.isfinite(col[o[0]]):
            continue
        i = o[0]; best, second = col[i], col[o[1]] if len(o) > 1 else np.inf
        mutual = np.argmin(D[i]) == j
        if mutual and best < MAXD and best < RATIO * second:
            L[b] = [w0s[i], round(float(best), 1), round(float(second), 1)]
    off = [int(b[1:]) - int(v[0][1:]) for b, v in L.items()]
    links[f"{b1}->{b0}"] = L
    report[f"{b1}->{b0}"] = {"slab": [round(za), round(zb)], "n1": len(w1s), "linked": len(L),
                             "offsets": {str(k): off.count(k) for k in sorted(set(off))}}
    print(f"{b1}->{b0} slab z {za:.0f}-{zb:.0f}: {len(L)}/{len(w1s)} linked, offsets {report[f'{b1}->{b0}']['offsets']}")
# chain into sheets: a sheet is a list of (band, winding) through consecutive bands
sheets, where = [], {}
for i, (b, _) in enumerate(bands):
    for w in sorted(G[b]):
        if i > 0:
            prev = links[f"{b}->{bands[i - 1][0]}"].get(w)
            if prev and (bands[i - 1][0], prev[0]) in where:
                s = where[(bands[i - 1][0], prev[0])]; sheets[s][b] = w; where[(b, w)] = s; continue
        where[(b, w)] = len(sheets); sheets.append({b: w})
json.dump({"sector": [x0, x1, y0, y1], "report": report, "links": links, "sheets": [s for s in sheets if len(s) > 1]},
          open(out, "w"), indent=1)
full = [s for s in sheets if len(s) == len(bands)]
print(len(full), "sheets run through all", len(bands), "bands;", sum(len(s) > 1 for s in sheets), "span two or more")
