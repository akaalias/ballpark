"""AGENT-EDITABLE — training procedure. One model per area (CLAUDE.md §1).

Current: map-cell classification with FULL-LATTICE coverage. Each EPOCH is one
complete pass over ALL train-lattice positions (not a random subsample), with a
fresh random rotation drawn per position per epoch, so over the run the model
sees epochs*len(crops) distinct (position, heading) vantages and every map cell
is visited many times — the coverage a C-way cell classifier needs to memorize
its box. Positions are streamed in shuffled chunks of CHUNK_SIZE so only one
chunk of pixels is ever resident. Targets are DECODE-CONSISTENT: the cell
target is a bilinear tent over the <=4 nearest cell centres (whose centroid is
the true position, matching what the 3x3 pooled decode reads), and the offset
target is the residual that pooled centroid leaves rather than the containing
cell's within-cell position. `--max-crops-per-bucket` is still accepted for
CLI compatibility but no longer controls sampling; it is logged as
`max_crops_per_bucket_arg`.

PAIRED-SURVEY INTERLEAVE (capture-holdout era): an area carries several
training photographs of the same ground. The baseline made one full lattice
pass per photograph in sequence, so each epoch fitted bb_dop20, then
overwrote it with truedop_2022, then with truedop_2023 — it forgot two of the
three captures every epoch (train diagnostics 1.53 / 1.40 / 0.015). Now each
chunk draws a set of lattice positions and extracts the crop at each position
from EVERY training capture under ONE shared heading, and the whole
(positions x captures) chunk is shuffled into mini-batches: no mini-batch is
single-survey and the epoch never ends on one photograph. Views per epoch are
unchanged (len(crops) * len(imgs)); per-capture top1 is logged.

NUISANCE RANDOMISATION: three photographs were absorbed as three pixel-exact
fingerprints (+20 brightness or a 1 px blur on a training photograph broke the
model). Every training view is now re-photographed at random on the device by
augment_views() — blur, exposure, per-channel colour gain, contrast,
brightness, gamma, saturation, sensor noise, drawn independently per view (AUG)
— so one survey's exposure/sharpness fingerprint no longer predicts the cell
and only structure shared across the photographs does. Training only; the
exported graph is unchanged.

VIEWS ARE CUT ON THE DEVICE (infrastructure, 2026-10-31, issue #3): the
training captures sit on the GPU as one uint8 tensor and gpu_crops() cuts and
rotates each mini-batch there (same geometry as pipeline.dataset.extract_crop,
within 1/255), and running stats stay on the device until the epoch ends.
This replaced one-at-a-time PIL rotation on the CPU and per-step GPU->CPU
reads: ~24% faster per epoch on the M1, more on a CUDA card. Keep new
per-view work on the device and avoid .item()/.cpu()/int()/float() on device
tensors inside the step loop — each one stalls the GPU queue.

Usage: python -m model.train --area berlin --out-dir runs/<id> [--epochs 2]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model.model import CELL_PX, build_model, export_onnx, grid_dims, loss_fn
from pipeline.common import DATA_DIR, area_dir, buckets, load_meta
from pipeline.dataset import CROP_PX, WINDOW_PX, list_crops

CHUNK_SIZE = 4096  # lattice positions extracted at a time (bounds memory)
BATCH_SIZE = 64
# Per-view photometric + sharpness nuisance ranges (training only).
AUG = dict(blur_p=0.7, blur_sigma_max=1.5, blur_taps=7, exposure=(0.75, 1.30),
           channel_gain=(0.85, 1.15), contrast=(0.6, 1.4),
           brightness=(-0.12, 0.12), gamma=(0.7, 1.4), saturation=(0.5, 1.5),
           noise_sigma_max=0.03)


def augment_views(x: torch.Tensor, aug: dict = AUG) -> torch.Tensor:
    """Re-photograph each view at random. x: float [B,3,H,W] in [0,1] on its
    device; every parameter is drawn independently per view. Blur first, so
    the noise added last is not blurred. Fully vectorised (MPS/CUDA/CPU)."""
    B, C, H, W = x.shape
    dev = x.device

    def U(lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(*shape, device=dev)

    # a. separable Gaussian blur, per-sample sigma (sigma~0 -> delta kernel)
    taps = aug["blur_taps"]; r = taps // 2
    sigma = U(0.0, aug["blur_sigma_max"], B)
    sigma = torch.where(torch.rand(B, device=dev) < aug["blur_p"], sigma,
                        torch.zeros_like(sigma))
    t = torch.arange(-r, r + 1, device=dev, dtype=x.dtype)
    k = torch.exp(-t.unsqueeze(0) ** 2 / (2 * sigma.clamp_min(0.05).unsqueeze(1) ** 2))
    k = k / k.sum(dim=1, keepdim=True)                       # [B, taps]
    k = k.repeat_interleave(C, dim=0)                        # [B*C, taps]
    y = x.reshape(1, B * C, H, W)
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"),
                 k.view(B * C, 1, taps, 1), groups=B * C)
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="reflect"),
                 k.view(B * C, 1, 1, taps), groups=B * C)
    x = y.reshape(B, C, H, W)
    # b. exposure gain x per-channel colour gain
    x = x * (U(*aug["exposure"], B, 1, 1, 1) * U(*aug["channel_gain"], B, C, 1, 1))
    # c. contrast about the crop's own mean
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = m + U(*aug["contrast"], B, 1, 1, 1) * (x - m)
    # d. brightness offset
    x = x + U(*aug["brightness"], B, 1, 1, 1)
    # e. gamma
    x = x.clamp(1e-4, 1) ** U(*aug["gamma"], B, 1, 1, 1)
    # f. saturation toward per-pixel grey
    grey = x.mean(dim=1, keepdim=True)
    x = grey + U(*aug["saturation"], B, 1, 1, 1) * (x - grey)
    # g. additive sensor noise
    x = x + torch.randn_like(x) * U(0.0, aug["noise_sigma_max"], B, 1, 1, 1)
    return x.clamp(0, 1)


def load_scene(area: str, data_dir: Path):
    """Meta + one image per TRAIN bucket (capture x lighting). The eval
    capture is never loaded here: the scorer asks about a photograph the
    model has not seen (pipeline/common.py "captures")."""
    meta = load_meta(area, data_dir)
    names = list(buckets(meta, "train"))
    imgs = [np.asarray(Image.open(area_dir(area, data_dir) / "relight" / f"{b}.png"))
            for b in names]
    print(f"[{area}] training buckets: {names}", flush=True)
    crops = list_crops(area, meta["width"], meta["height"], "train")
    return meta, names, imgs, crops


def chunk_views(meta, crop_xy, indices, rng, n_caps):
    """Index one chunk of PAIRED vantages: every position in `indices` in
    every capture, with one shared heading per position. No pixels are cut
    here — gpu_crops() does that per mini-batch. Returns CPU tensors
    (cx [N*K], cy [N*K], angle_deg [N*K], cell_idx [N*K], pos [N*K,2],
    cap_idx [N*K]), views ordered position-major as before.

    `cell_idx` is the containing cell, kept only for the top1 log line — the
    loss targets are built from `pos`, the continuous raster-pixel position.
    """
    gw, _ = grid_dims(meta)
    xy = crop_xy[torch.as_tensor(indices, dtype=torch.int64)]          # [N,2]
    ang = torch.as_tensor(rng.uniform(0, 360, size=len(indices)),
                          dtype=torch.float32)                         # shared heading
    rep = lambda t: t.repeat_interleave(n_caps, dim=0)
    cx, cy = rep(xy[:, 0]), rep(xy[:, 1])
    cells = (cy // CELL_PX) * gw + (cx // CELL_PX)
    pos = torch.stack([cx, cy], dim=1).float()
    caps = torch.arange(n_caps, dtype=torch.int64).repeat(len(indices))
    return cx, cy, rep(ang), cells, pos, caps


def gpu_crops(scene_u8, cap, cx, cy, angle_deg, size=CROP_PX):
    """Cut and rotate a batch of training views on the device. Same geometry
    as pipeline.dataset.extract_crop (PIL rotate, bilinear, about the crop
    centre), checked to match it within uint8 rounding at the edges.

    scene_u8: [K,3,H,W] uint8 (all training captures, resident on device).
    cap, cx, cy: int64 [B]; angle_deg: float [B]; all on the same device.
    Returns float [B,3,size,size] in [0,1].
    """
    dev = scene_u8.device
    hw = WINDOW_PX // 2 + 1                  # extract_crop's half window
    ar = torch.arange(2 * hw, device=dev)
    ys = (cy - hw).unsqueeze(1) + ar         # [B,2hw]
    xs = (cx - hw).unsqueeze(1) + ar
    win = scene_u8[cap[:, None, None], :, ys[:, :, None], xs[:, None, :]]   # [B,2hw,2hw,3]
    win = win.permute(0, 3, 1, 2).float()
    # PIL: input = R(-angle) applied to (output - centre) + centre, in pixel-
    # corner coordinates; the window centre is hw and the crop is centred on it.
    th = torch.deg2rad(angle_deg.float())
    c, s = torch.cos(th), torch.sin(th)
    d = torch.arange(size, device=dev, dtype=torch.float32) + 0.5 - size / 2
    dx, dy = d.view(1, 1, size), d.view(1, size, 1)
    px = hw + c.view(-1, 1, 1) * dx - s.view(-1, 1, 1) * dy
    py = hw + s.view(-1, 1, 1) * dx + c.view(-1, 1, 1) * dy
    grid = torch.stack([px, py], dim=-1) * (2.0 / (2 * hw)) - 1.0           # [B,S,S,2]
    out = F.grid_sample(win, grid, mode="bilinear", padding_mode="zeros",
                        align_corners=False)
    return out.div_(255.0)


def make_soft_targets(pos, meta, n_cells, gw, gh, smoothing=0.05):
    """Bilinear tent target over the <=4 cell centres nearest each true
    position. pos: [B,2] raster px. Returns dense [B, n_cells] float32.

    The decode reads a 3x3-pooled centroid, so the target is the distribution
    whose centroid IS the true position: nearly identical border views now get
    nearly identical targets instead of contradictory hard labels. Index
    clamping merges out-of-grid neighbours' mass into the border cell via
    scatter_add_ — that mass is kept, not dropped, so rows always sum to 1.
    """
    fu = pos[:, 0] / CELL_PX - 0.5   # position in cell-centre coordinates
    fv = pos[:, 1] / CELL_PX - 0.5
    i0 = fu.floor(); j0 = fv.floor()
    a = (fu - i0); b = (fv - j0)     # in [0,1)
    tgt = torch.zeros(len(pos), n_cells, dtype=torch.float32, device=pos.device)
    for di, wu in ((0, 1 - a), (1, a)):
        for dj, wv in ((0, 1 - b), (1, b)):
            ci = (i0 + di).clamp(0, gw - 1).long()
            cj = (j0 + dj).clamp(0, gh - 1).long()
            tgt.scatter_add_(1, (cj * gw + ci).unsqueeze(1), (wu * wv).unsqueeze(1))
    return tgt * (1 - smoothing) + smoothing / n_cells


def residual_offset_target(model, logits, pos, meta):
    """(true position - model's 3x3 pooled centroid), in cell units,
    clamped to the offset head's (-0.5, 0.5) range. Detached.

    Mirrors forward()'s pooling exactly, using the model's own buffers so the
    grid geometry stays single-sourced. Early in training the argmax is noisy,
    so this target is nonstationary; that is expected and bounded by the clamp.
    """
    with torch.no_grad():
        p = F.softmax(logits, dim=1)
        idx = p.argmax(dim=1)
        win_u = model.cell_u.index_select(0, idx)
        win_v = model.cell_v.index_select(0, idx)
        du = (model.cell_u.unsqueeze(0) - win_u.unsqueeze(1)).abs()
        dv = (model.cell_v.unsqueeze(0) - win_v.unsqueeze(1)).abs()
        mask = ((du <= 1.6 * model.scale_u) & (dv <= 1.6 * model.scale_v)).float()
        q = p * mask
        m = q.sum(dim=1).clamp_min(1e-9)
        cu = (q * model.cell_u.unsqueeze(0)).sum(dim=1) / m
        cv = (q * model.cell_v.unsqueeze(0)).sum(dim=1) / m
        true_u = pos[:, 0] / meta["width"]; true_v = pos[:, 1] / meta["height"]
        ru = ((true_u - cu) / model.scale_u).clamp(-0.5, 0.5)
        rv = ((true_v - cv) / model.scale_v).clamp(-0.5, 0.5)
        return torch.stack([ru, rv], dim=1)


def run_epoch(model, opt, meta, scene, crop_xy, rng, device, n_caps,
              pos_per_chunk, gw, gh, max_chunks=None):
    """One full pass over the whole lattice x every capture, shuffled, fresh
    headings; each chunk holds its positions in ALL captures. Running stats
    stay on the device and are read once at the end, so the CPU never waits
    for the GPU mid-epoch. `max_chunks` is for timing runs only."""
    model.train()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.int64, device=device)
    conf_sum = torch.zeros((), device=device)
    cap_correct = torch.zeros(n_caps, dtype=torch.int64, device=device)
    cap_seen = torch.zeros(n_caps, dtype=torch.int64, device=device)
    seen = 0
    order = rng.permutation(len(crop_xy))
    for ci, start in enumerate(range(0, len(order), pos_per_chunk)):
        if max_chunks is not None and ci >= max_chunks:
            break
        views = chunk_views(meta, crop_xy, order[start:start + pos_per_chunk],
                            rng, n_caps)
        cx, cy, ang, cell_idx, pos, cap_idx = (t.to(device) for t in views)
        n = len(cx)
        perm = torch.randperm(n).to(device)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            B = len(idx)
            cap_b = cap_idx[idx]
            xb = augment_views(gpu_crops(scene, cap_b, cx[idx], cy[idx], ang[idx]))
            cb, pb = cell_idx[idx], pos[idx]
            logits, off_pred = model.training_outputs(xb)
            soft_t = make_soft_targets(pb, meta, model.n_cells, gw, gh)
            off_t = residual_offset_target(model, logits, pb, meta)
            loss = loss_fn(logits, off_pred, soft_t, off_t)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                loss_sum += loss.detach() * B
                # top1 is measured on AUGMENTED views, so it reads lower
                # than experiment 2's 0.79 by design; judge by eval.
                # top1 is vs. the CONTAINING cell, so it now reads lower
                # than exp 4's 0.83 BY DESIGN: border views are supposed
                # to split mass across the tent's cells. Judge
                # convergence by loss and by eval, not by this.
                hit = (logits.argmax(dim=1) == cb).long()
                correct += hit.sum()
                cap_correct.scatter_add_(0, cap_b, hit)
                cap_seen.scatter_add_(0, cap_b, torch.ones_like(cap_b))
                conf_sum += F.softmax(logits, dim=1).max(dim=1).values.sum()
                seen += B
    return {"loss_sum": float(loss_sum), "correct": int(correct),
            "conf_sum": float(conf_sum), "seen": seen,
            "cap_correct": cap_correct.cpu().tolist(),
            "cap_seen": cap_seen.cpu().tolist()}


def train_area(area: str, out_dir: Path, data_dir: Path, epochs: int,
               max_crops_per_bucket: int, seed: int) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    meta, names, imgs, crops = load_scene(area, data_dir)
    gw, gh = grid_dims(meta)
    n_caps = len(imgs)
    pos_per_chunk = max(1, CHUNK_SIZE // n_caps)
    model = build_model(meta).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # A 12x-longer schedule wants a decaying LR, not the constant 1e-3.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                       eta_min=1e-4)
    print(f"[{area}] {len(crops)} lattice positions, {model.n_cells} cells, "
          f"device={device}")
    # All training captures live on the device for the whole run (uint8,
    # ~0.43 GB for Berlin); views are cut from them per mini-batch.
    scene = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous().to(device)
    del imgs
    crop_xy = torch.tensor([(c["cx"], c["cy"]) for c in crops], dtype=torch.int64)
    final_top1 = 0.0
    final_top1_by_cap = {nm: 0.0 for nm in names}
    for epoch in range(epochs):
        stats = run_epoch(model, opt, meta, scene, crop_xy, rng, device,
                          n_caps, pos_per_chunk, gw, gh)
        sched.step()
        seen = stats["seen"]
        final_top1 = stats["correct"] / max(seen, 1)
        final_top1_by_cap = {
            nm: round(stats["cap_correct"][k] / max(stats["cap_seen"][k], 1), 4)
            for k, nm in enumerate(names)}
        print(f"[{area}] epoch {epoch + 1}/{epochs} loss={stats['loss_sum'] / max(seen, 1):.4f} "
              f"top1={final_top1:.4f} mean_max_softmax={stats['conf_sum'] / max(seen, 1):.4f} "
              f"views={seen} lr={sched.get_last_lr()[0]:.2e} "
              f"top1_by_capture={final_top1_by_cap}", flush=True)

    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = models_dir / f"{area}.onnx"
    export_onnx(model.cpu(), str(onnx_path))
    return {
        "area": area,
        "n_train_crops": epochs * len(crops) * n_caps,
        "epochs": epochs,
        "device": device,
        "train_seconds": round(time.time() - t0, 1),
        "onnx_bytes": onnx_path.stat().st_size,
        "init": "from-scratch",  # §9: log init strategy per experiment
        "n_cells": model.n_cells,
        "distinct_vantages_seen": epochs * len(crops) * n_caps,
        "full_lattice_epochs": True,
        "final_train_top1": round(final_top1, 4),
        "chunk_size": CHUNK_SIZE,
        "max_crops_per_bucket_arg": max_crops_per_bucket,
        "capture_sampling": ("paired-position interleave, shared heading per position, "
                             "per-view photometric+sharpness nuisance randomisation"),
        "augmentation": AUG,
        "captures": names,
        "positions_per_chunk": pos_per_chunk,
        "final_train_top1_by_capture": final_top1_by_cap,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-crops-per-bucket", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    out_dir = Path(args.out_dir)
    info = train_area(args.area, out_dir, data_dir, args.epochs,
                      args.max_crops_per_bucket, args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_log = out_dir / "train_info.json"
    logs = json.loads(train_log.read_text()) if train_log.exists() else []
    logs.append(info)
    train_log.write_text(json.dumps(logs, indent=2))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
