"""selftrain.py: self-training of one ink model on rendered sheet surfaces (after the 2023 pseudo-label loop).
Needs villa (github.com/ScrollPrize/villa): set KOINE to its ink-detection directory and VESUVIUS_SRC to vesuvius/src.
Inputs per item (an e98_look output dir with SAVE_SV=1): sv.npy (layers, H, W), maps/{model}_rev_dXX.png.
Labels: z-scored mean of every saved model map over DEPTHS_L, smoothed; top POSQ quantile = ink, below NEGQ = not ink,
the rest ignored. Fine-tune MODEL on TRAIN items only, then predict every item at DEPTHS so a held-out item shows
whether the tuned model is sharper than the base one. Usage: e110_selftrain.py RUN_DIR OUT_DIR
Env: MODEL reader2|dnative|ink42|v3B (v3B: path in V3B), TRAIN "c4,c6", ITEMS "c3,c4,c6", DEPTHS "-6,-3,0,3,6", DEPTHS_L "-3,0,3",
     STEPS 2500, LR 2e-5, POSQ 0.97, NEGQ 0.70, ROUNDS 1 (later rounds relabel from the tuned model's own maps),
     INIT (a saved ft_MODEL.pt; with STEPS 0 the script only predicts ITEMS with those weights)."""
import sys, os, glob, re, json, time
sys.path[:0] = [os.environ["KOINE"], os.environ["VESUVIUS_SRC"]]
import numpy as np, torch, imageio.v2 as imageio
from scipy import ndimage as ndi
from koine_machines.inference.infer import build_repo_training_model_bundle
from vesuvius.image_proc.intensity.normalization import normalize_robust

run, out = sys.argv[1], sys.argv[2]
E = os.environ.get
MODEL = E("MODEL", "reader2"); TRAIN = E("TRAIN", "c4,c6").split(","); ITEMS = E("ITEMS", "c3,c4,c6").split(",")
DEPTHS = [int(x) for x in E("DEPTHS", "-6,-3,0,3,6").split(",")]; DL = [int(x) for x in E("DEPTHS_L", "-3,0,3").split(",")]
STEPS, LR, POSQ, NEGQ, ROUNDS = int(E("STEPS", "2500")), float(E("LR", "2e-5")), float(E("POSQ", "0.97")), float(E("NEGQ", "0.70")), int(E("ROUNDS", "1"))
P, Z, B = 128, 17, 16
dev = "cuda"; T0 = time.time(); rng = np.random.default_rng(0)
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
HF = "https://huggingface.co"
CK = {"ink42": f"{HF}/scrollprize/ink_9um/resolve/main/hybrid_3d2d-seed42/step-075000.pth",
      "reader2": f"{HF}/domenicor046/reader-v2/resolve/main/reader-v2-step040000.pth"}


def load_net():
    import subprocess, json as js
    p = f"/tmp/{MODEL}.pth"
    if MODEL == "v3B":                                                      # TRACE v3B, our own (private dataset)
        p = E("V3B")
    elif MODEL == "dnative" and not os.path.exists(p):
        from safetensors.torch import load_file
        b = f"{HF}/Nieuwlaar/ink9um-dense-native/resolve/main"
        subprocess.run(["curl", "-sSL", "--retry", "5", "-o", "/tmp/dn.st", f"{b}/weights/dense_native_016000.safetensors"], check=True)
        subprocess.run(["curl", "-sSL", "--retry", "5", "-o", "/tmp/dn.json", f"{b}/training/configs/train_dense_native.json"], check=True)
        torch.save({"model": load_file("/tmp/dn.st"), "config": js.load(open("/tmp/dn.json")), "step": 16000}, p)
    elif not os.path.exists(p):
        subprocess.run(["curl", "-sSL", "--retry", "5", "-o", p, CK[MODEL]], check=True)
    return build_repo_training_model_bundle(torch.load(p, map_location="cpu", weights_only=False), p).model.to(dev)


def labels(item, prefix, mdir):
    """Ternary label map from the saved maps of an item: 1 ink, 0 not ink, -1 ignore."""
    fs = [f for f in glob.glob(f"{mdir}/{item}/maps/{prefix}*_rev_d*.png") if int(re.search(r"_d([+-]\d+)", f).group(1)) in DL]
    by = {}
    for f in fs:
        by.setdefault(os.path.basename(f).split("_rev_")[0], []).append(imageio.imread(f).astype(np.float32) / 255)
    valid = SV[item][SV[item].shape[0] // 2] > 0
    zs = []
    for k, ms in by.items():
        m = np.mean(ms, 0); mu, sd = m[valid].mean(), m[valid].std() + 1e-6; zs.append((m - mu) / sd)
    e = ndi.gaussian_filter(np.mean(zs, 0), 3)
    hi, lo = np.quantile(e[valid], POSQ), np.quantile(e[valid], NEGQ)
    lab = np.full(e.shape, -1, np.int8); lab[e >= hi] = 1; lab[e <= lo] = 0; lab[~valid] = -1
    log(item, "label models", sorted(by), "ink frac", round(float((lab == 1).mean()), 4))
    return lab


def window(sv, d, r, q):
    c = sv.shape[0] // 2
    return sv[c + d - 8:c + d + 9, r:r + P, q:q + P][::-1]                 # rev, as e98_look


def predict(net, sv, d, stride=64):
    net.eval(); H, Wd = sv.shape[1:]
    o = np.zeros((H, Wd), np.float32); wt = np.zeros((H, Wd), np.float32)
    w1 = np.hanning(P).astype(np.float32); w2 = np.maximum(np.outer(w1, w1), 1e-3)
    co = [(r, q) for r in range(0, max(H - P, 0) + 1, stride) for q in range(0, max(Wd - P, 0) + 1, stride)]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for k in range(0, len(co), 64):
            b = co[k:k + 64]; x = np.stack([window(sv, d, r, q) for r, q in b]).astype(np.float32)
            keep = (x[:, Z // 2] > 0).reshape(len(b), -1).mean(1) > 0.3
            if not keep.any():
                continue
            b = [bb for bb, kk in zip(b, keep) if kk]
            x = np.stack([normalize_robust(p) for p in x[keep]]).astype(np.float32)
            pr = torch.sigmoid(net(torch.from_numpy(x)[:, None].to(dev)).float())[:, 0].cpu().numpy()
            for (r, q), pp in zip(b, pr):
                o[r:r + P, q:q + P] += pp * w2; wt[r:r + P, q:q + P] += w2
    return np.where(wt > 0, o / np.maximum(wt, 1e-6), 0)


SV = {it: np.load(f"{run}/{it}/sv.npy", mmap_mode="r") for it in ITEMS}
net = load_net(); labels_r = {}
if E("INIT"):
    net.load_state_dict(torch.load(E("INIT"), map_location=dev)); log("weights", E("INIT"))
if STEPS and not glob.glob(f"{run}/{ITEMS[0]}/maps/{MODEL}_rev_d*.png"):          # untuned maps for comparison
    for it in ITEMS:
        os.makedirs(f"{out}/{it}/maps", exist_ok=True)
        for d in DEPTHS:
            imageio.imwrite(f"{out}/{it}/maps/base_{MODEL}_rev_d{d:+03d}.png", (np.clip(predict(net, SV[it], d), 0, 1) * 255).astype(np.uint8))
    log("base maps done")
for rnd in range(ROUNDS):
    if not STEPS:
        TRAIN = []
    LAB = {it: labels(it, "", run) if rnd == 0 else labels_r[it] for it in TRAIN}
    pos = {it: np.argwhere(LAB[it] == 1) for it in TRAIN}; anyv = {it: np.argwhere(LAB[it] >= 0) for it in TRAIN}
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4); scaler = torch.cuda.amp.GradScaler()
    net.train(); run_loss = []
    for step in range(STEPS if TRAIN else 0):
        xs, ys = [], []
        while len(xs) < B:
            it = TRAIN[rng.integers(len(TRAIN))]; sv, lab = SV[it], LAB[it]
            src = pos[it] if rng.random() < 0.5 and len(pos[it]) else anyv[it]
            r, q = src[rng.integers(len(src))] - P // 2 + rng.integers(-32, 33, 2)
            r = int(np.clip(r, 0, sv.shape[1] - P)); q = int(np.clip(q, 0, sv.shape[2] - P))
            y = lab[r:r + P, q:q + P]
            if (y >= 0).mean() < 0.3:
                continue
            x = window(sv, int(rng.choice(DL)), r, q).astype(np.float32)
            k = int(rng.integers(4)); x = np.rot90(x, k, (1, 2)); y = np.rot90(y, k)
            if rng.random() < 0.5:
                x = x[:, :, ::-1]; y = y[:, ::-1]
            xs.append(normalize_robust(np.ascontiguousarray(x)).astype(np.float32)); ys.append(np.ascontiguousarray(y))
        x = torch.from_numpy(np.stack(xs))[:, None].to(dev); y = torch.from_numpy(np.stack(ys).astype(np.float32)).to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            lo = net(x).float()[:, 0]
        m = (y >= 0).float()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(lo, y.clamp(0, 1), reduction="none") * m).sum() / m.sum()
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); run_loss.append(loss.item())
        if step % 250 == 0 or step == STEPS - 1:
            log(f"round {rnd} step {step} loss {np.mean(run_loss[-250:]):.4f}")
    for it in ITEMS:
        os.makedirs(f"{out}/{it}/maps", exist_ok=True)
        for d in sorted(set(DEPTHS) | set(DL)):
            p = predict(net, SV[it], d)
            imageio.imwrite(f"{out}/{it}/maps/ft{rnd}_{MODEL}_rev_d{d:+03d}.png", (np.clip(p, 0, 1) * 255).astype(np.uint8))
        log(it, "round", rnd, "maps done")
    if rnd + 1 < ROUNDS:                                                    # relabel from the tuned model alone
        labels_r = {it: labels(it, f"ft{rnd}_", out) for it in TRAIN}
if STEPS:
    torch.save(net.state_dict(), f"{out}/ft_{MODEL}.pt")
log("finished", out)
