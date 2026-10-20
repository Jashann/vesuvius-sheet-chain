# dupcheck.py DIR_A/ITEM DIR_B/ITEM [x0 x1] : are two rendered windings the same physical sheet? Snapping can pull
# two fitted windings onto one sheet, so the same text then shows up "on two windings". Raw CT texture (fibres,
# cracks) never repeats on separate sheets, so for 160 px tiles of A (optionally within columns x0..x1) we search
# B for the best normalised cross-correlation of CT d=0 (dx +-600, dy +-60), in place and around the three best global
# shifts (FFT on 4x-downsampled CT over every offset), so a repeat one full turn away or at another z is caught too.
# Tiles with NCC > 0.4 at a common shift (+-40 px, +-20 px) are counted as duplicated. Prints the duplicated fraction and the global shift.
import sys, numpy as np
from PIL import Image
from scipy.signal import fftconvolve
def ct(p): return np.asarray(Image.open(f"{p}/ct/d+00.jpg").convert("L"), np.float32)
A, B = ct(sys.argv[1]), ct(sys.argv[2]); x0, x1 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, A.shape[1])
def ncc_map(t, im):
    t = t - t.mean(); n = np.sqrt((t * t).sum()) + 1e-6
    num = fftconvolve(im, t[::-1, ::-1], mode="valid")
    s1 = fftconvolve(im, np.ones_like(t), mode="valid"); s2 = fftconvolve(im * im, np.ones_like(t), mode="valid")
    var = np.maximum(s2 - s1 * s1 / t.size, 1e-6)
    return np.where(var > 0.05 * np.median(var), num / (n * np.sqrt(var)), -1)
def gshift(A, B, f=4, minfrac=0.15, k=3):
    """Masked normalised cross-correlation of high-passed, f-times downsampled CT over every (dx, dy), by FFT.
    Returns the k best peaks (corr, dx, dy), B's matching position = A's position - (dx, dy)."""
    from scipy.ndimage import gaussian_filter, maximum_filter
    def prep(X):
        x = X[::f, ::f].astype(np.float64); m = (x > 0).astype(np.float64)
        x = x - gaussian_filter(x, 3) / np.maximum(gaussian_filter(m, 3), 1e-3); x *= m
        return x / (x[m > 0].std() + 1e-6), m
    a, ma = prep(A); b, mb = prep(B)
    xc = lambda x, y: fftconvolve(x, y[::-1, ::-1], mode="full")
    num, na, nb, cnt = xc(a, b), xc(a * a, mb), xc(ma, b * b), xc(ma, mb)
    ok = cnt > minfrac * min(ma.sum(), mb.sum())
    c = np.where(ok, num / np.sqrt(np.maximum(na * nb, 1e-6)), -1)
    pk = np.argwhere((c == maximum_filter(c, 60)) & (c > 0))
    pk = sorted(pk, key=lambda q: -c[q[0], q[1]])[:k]
    return [(float(c[i, j]), (j - (b.shape[1] - 1)) * f, (i - (b.shape[0] - 1)) * f) for i, j in pk]


def tiles(gdx, gdy, T=160):
    res = []
    for y in range(0, A.shape[0] - T, T):
        for x in range(x0, min(x1, A.shape[1]) - T, T):
            t = A[y:y + T, x:x + T]
            if (t > 0).mean() < 0.9: continue
            ya, yb = max(0, y - gdy - 60), min(B.shape[0], y - gdy + T + 60); xa, xb = max(0, x - gdx - 600), min(B.shape[1], x - gdx + T + 600)
            win = B[ya:yb, xa:xb]
            if win.shape[0] < T or win.shape[1] < T or (win > 0).mean() < 0.7: continue
            m = ncc_map(t, win); k = np.unravel_index(np.argmax(m), m.shape)
            res.append((float(m[k]), xa + k[1] - x, ya + k[0] - y))
    return np.array(res)


def consistent(r):                                  # NCC > 0.4 at (about) the common shift: real repeats agree, noise scatters
    h = r[:, 0] > 0.4
    if not h.any(): return h
    mx, my = np.median(r[h, 1]), np.median(r[h, 2])
    return h & (np.abs(r[:, 1] - mx) < 40) & (np.abs(r[:, 2] - my) < 20)


best = None                                          # tile search in place and around the top global peaks; keep the best
for gc, gdx, gdy in [(0.0, 0, 0)] + gshift(A, B):
    r = tiles(gdx, gdy)
    if len(r) >= 30 and (best is None or consistent(r).mean() > consistent(best[0]).mean()):
        best = (r, gc, gdx, gdy)
if best is None: print("no overlap"); sys.exit()
r, gc, gdx, gdy = best
dup = consistent(r)
print(f"global shift dx {gdx} dy {gdy} (CT corr {gc:.2f})")
print(f"{sys.argv[1]} vs {sys.argv[2]}: tiles {len(r)}, duplicated {dup.mean():.2f}, median best NCC {np.median(r[:,0]):.2f}",
      f"shift dx {np.median(r[dup,1]) if dup.any() else float('nan'):.0f} dy {np.median(r[dup,2]) if dup.any() else float('nan'):.0f}")
