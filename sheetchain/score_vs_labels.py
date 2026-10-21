"""Score ink maps against human ink labels: AUC, average precision and the share of background marked as ink at 80%
ink recall (the false-ink measure of the Reader V3 README). Pixels count only inside the certainty mask and where the
CT layer is non-zero, at least MARGIN px (default 200) from the edge of the render so every tile size covers them.
Usage: python score_vs_labels.py CT_LAYER.tif LABELS.png MASK.png MAP.png [MAP.png ...]
Maps are uint8 ink probability x 255 on the same pixel grid as the labels (label > 127 = ink)."""
import os, sys, json
import numpy as np, imageio.v2 as imageio
from scipy import ndimage as ndi


def gray(f):
    a = imageio.imread(f)
    return a if a.ndim == 2 else a[..., 0]


def scores(p, y, m):
    hp = np.bincount(p[m & y], minlength=256).astype(np.float64); hn = np.bincount(p[m & ~y], minlength=256).astype(np.float64)
    np_, nn = hp.sum(), hn.sum()
    below = np.concatenate([[0], np.cumsum(hn)[:-1]])
    auc = float((hp * below + 0.5 * hp * hn).sum() / (np_ * nn))
    tp, fp = np.cumsum(hp[::-1]), np.cumsum(hn[::-1])                     # thresholds from high to low
    rec, prec = tp / np_, tp / np.maximum(tp + fp, 1)
    ap = float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))
    k = int(np.argmax(rec >= 0.8))
    return dict(auc=round(auc, 4), ap=round(ap, 4), bg_at_80=round(float(fp[k] / nn), 4))


ct, lab, mask, maps = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
margin = int(os.environ.get("MARGIN", "200"))
valid = gray(ct) > 0
inner = ndi.zoom((ndi.distance_transform_edt(valid[::4, ::4]) * 4 >= margin).astype(np.uint8), 4, order=0)
inner = np.pad(inner, [(0, max(0, valid.shape[0] - inner.shape[0])), (0, max(0, valid.shape[1] - inner.shape[1]))])[:valid.shape[0], :valid.shape[1]] > 0
y = gray(lab) > 127; m = (gray(mask) > 0) & valid & inner
print(json.dumps({"pixels": int(m.sum()), "ink_frac": round(float(y[m].mean()), 4)}))
for f in maps:
    print(os.path.basename(f), json.dumps(scores(gray(f), y, m)))
