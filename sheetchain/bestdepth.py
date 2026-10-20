# bestdepth.py DIR ITEM OUT.png [TILE] : per-patch best depth for one rendered item (maps/<model>_rev_d<+dd>.png).
# Mean of the three models' z-scores per depth; in each TILE x TILE patch pick the depth whose mean map has the
# highest local contrast (p99 - p50), smooth the chosen depth field, then average each pixel over a soft (gaussian)
# depth window around it. Corrects slow surface misfit that blurs letters at a single fixed depth. Also writes OUT_depth.png.
import sys, glob, re, numpy as np
from PIL import Image
from scipy import ndimage as ndi
D, it, O = sys.argv[1:4]; T = int(sys.argv[4]) if len(sys.argv) > 4 else 256
fs = glob.glob(f"{D}/{it}/maps/*_rev_d*.png")
ds = sorted({int(re.search(r"_d([+-]\d+)", f).group(1)) for f in fs}); ms = sorted({f.split("/")[-1].split("_rev")[0] for f in fs})
def z(m, d):
    a = np.asarray(Image.open(f"{D}/{it}/maps/{m}_rev_d{d:+03d}.png").convert("L"), np.float32); return a
A = {m: np.stack([z(m, d) for d in ds]) for m in ms}
valid = A[ms[0]][len(ds) // 2] > 0
Z = np.zeros_like(A[ms[0]])
for m in ms:
    v = A[m][:, valid]; Z += (A[m] - v.mean()) / (v.std() + 1e-6)
Z /= len(ms)                                              # (n_depths, H, W)
H, W = valid.shape; gy, gx = (H + T - 1) // T, (W + T - 1) // T
score = np.full((len(ds), gy, gx), -np.inf, np.float32)
for i in range(gy):
    for j in range(gx):
        sl = (slice(i * T, (i + 1) * T), slice(j * T, (j + 1) * T)); m = valid[sl]
        if m.mean() < 0.3: continue
        for k in range(len(ds)):
            v = Z[k][sl][m]; score[k, i, j] = np.percentile(v, 99) - np.percentile(v, 50)
best = np.where(np.isfinite(score.max(0)), score.argmax(0), len(ds) // 2).astype(np.float32)
best = ndi.median_filter(best, 3)
field = ndi.zoom(ndi.gaussian_filter(best, 0.7), (H / gy, W / gx), order=1)[:H, :W]
field = np.pad(field, ((0, H - field.shape[0]), (0, W - field.shape[1])), mode="edge")
out = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
for k in range(len(ds)):                                  # soft window around the chosen depth, no seams
    w = np.exp(-0.5 * ((k - field) / 0.9) ** 2); out += w * Z[k]; cnt += w
out /= cnt; v = out[valid]; lo, hi = np.percentile(v, [2, 99.7])
img = (np.clip((out - lo) / (hi - lo), 0, 1) * valid * 255).astype(np.uint8)
Image.fromarray(img).save(O)
Image.fromarray((field / max(len(ds) - 1, 1) * 255).astype(np.uint8)).save(O.replace(".png", "_depth.png"))
print(it, "depths", ds, "models", ms, "chosen depth counts", np.bincount(np.round(best).astype(int).ravel(), minlength=len(ds)).tolist())
