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

CROSS-SURVEY BLEND: the three training photographs are pixel-aligned on one
grid, and every chunk already holds each position in all three under one
shared heading. Before augment_views(), each training view is replaced with
probability MIX['p'] by a per-pixel convex combination of its position's three
sibling crops, weighted by a smooth random field (MIX['grid']^2 Gaussian
logits x per-view temperature, bilinearly upsampled, softmax over captures) —
so canopy, shadow, movable objects and colour pipeline become locally
uninformative while geometry shared by all surveys stays sharp. Labels are
unchanged (same place). Training only; the exported graph is unchanged.

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
from pipeline.dataset import extract_crop, list_crops

CHUNK_SIZE = 4096  # lattice positions extracted at a time (bounds memory)
BATCH_SIZE = 64
# Per-view photometric + sharpness nuisance ranges (training only).
AUG = dict(blur_p=0.7, blur_sigma_max=1.5, blur_taps=7, exposure=(0.75, 1.30),
           channel_gain=(0.85, 1.15), contrast=(0.6, 1.4),
           brightness=(-0.12, 0.12), gamma=(0.7, 1.4), saturation=(0.5, 1.5),
           noise_sigma_max=0.03)
MIX = dict(p=0.75, grid=4, tau=(0.5, 3.0))
# p: probability a training view is replaced by a cross-survey blend;
# grid: edge of the low-res weight-logit grid (4 -> ~32 m blobs);
# tau: per-view logit scale range (small = soft everywhere-blend, large = patchwork of surveys).


def cross_survey_blend(x_u8: torch.Tensor, idx: torch.Tensor, cap_idx: torch.Tensor,
                       n_caps: int, device, mix: dict = MIX):
    """Replace each view (w.p. mix['p']) by a per-pixel convex blend of its
    position's sibling crops across all training captures.

    x_u8 is extract_chunk's uint8 [N*K,3,H,W], POSITION-MAJOR: view j belongs
    to position j // K and capture j % K. Returns (xb float [B,3,H,W] in [0,1]
    on device, mixed bool [B] on device, mean_max_w float over mixed views).
    """
    B = len(idx)
    H, W_ = x_u8.shape[-2], x_u8.shape[-1]
    if n_caps < 2:
        own = x_u8[idx].to(device).float().div_(255.0)
        return own, torch.zeros(B, dtype=torch.bool, device=device), 0.0
    p_idx = idx // n_caps
    sib = (p_idx.unsqueeze(1) * n_caps + torch.arange(n_caps)).reshape(-1)  # [B*K]
    S = x_u8[sib].to(device).float().div_(255.0).view(B, n_caps, 3, H, W_)
    own = S[torch.arange(B, device=device), cap_idx[idx].to(device)]  # [B,3,H,W]
    g = mix["grid"]
    lo, hi = mix["tau"]
    tau = lo + (hi - lo) * torch.rand(B, 1, 1, 1, device=device)
    z = torch.randn(B, n_caps, g, g, device=device) * tau
    z = F.interpolate(z, size=(H, W_), mode="bilinear", align_corners=False)
    Wt = torch.softmax(z, dim=1).unsqueeze(2)  # [B,K,1,H,W], sums to 1 over K
    blended = (S * Wt).sum(dim=1)              # [B,3,H,W]
    mixed = torch.rand(B, device=device) < mix["p"]
    xb = torch.where(mixed.view(B, 1, 1, 1), blended, own)
    if bool(mixed.any()):
        mmw = float(Wt.squeeze(2).max(dim=1).values.mean(dim=(1, 2))[mixed].mean())
    else:
        mmw = 0.0
    return xb, mixed, mmw


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


def extract_chunk(meta, imgs, crops, indices, rng):
    """Extract one chunk of PAIRED vantages: every position in `indices` from
    every capture in `imgs`, with one shared heading per position. Returns
    (x_uint8 [N*K,3,H,W], cell_idx [N*K], pos [N*K,2], cap_idx [N*K]).

    `cell_idx` is the containing cell, kept only for the top1 log line — the
    loss targets are built from `pos`, the continuous raster-pixel position.
    """
    gw, _ = grid_dims(meta)
    xs, cells, pos, caps = [], [], [], []
    for i in indices:
        c = crops[i]
        cx, cy = c["cx"], c["cy"]
        angle = float(rng.uniform(0, 360))  # heading augmentation, shared
        cell = (cy // CELL_PX) * gw + (cx // CELL_PX)
        for k, img in enumerate(imgs):
            xs.append(extract_crop(img, cx, cy, angle))
            cells.append(cell)
            pos.append((float(cx), float(cy)))
            caps.append(k)
    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).contiguous()
    return (x, torch.tensor(cells, dtype=torch.int64),
            torch.tensor(pos, dtype=torch.float32),
            torch.tensor(caps, dtype=torch.int64))


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
    final_top1 = 0.0
    final_top1_by_cap = {nm: 0.0 for nm in names}
    blend_frac = top1_blend = top1_unmixed = mmw_epoch = 0.0
    for epoch in range(epochs):
        model.train()
        loss_sum, correct, conf_sum, seen = 0.0, 0, 0.0, 0
        n_mixed, correct_mixed, correct_unmixed = 0, 0, 0
        mmw_sum, mmw_batches = 0.0, 0
        cap_correct = torch.zeros(n_caps, dtype=torch.int64)
        cap_seen = torch.zeros(n_caps, dtype=torch.int64)
        # One full pass over the whole lattice x every capture, shuffled, fresh
        # rotations; each chunk holds its positions in ALL captures.
        order = rng.permutation(len(crops))
        for start in range(0, len(order), pos_per_chunk):
            x_u8, cell_idx, pos, cap_idx = extract_chunk(
                meta, imgs, crops, order[start:start + pos_per_chunk], rng)
            n = len(x_u8)
            perm = torch.randperm(n)
            for i in range(0, n, BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                xb, mixed, mmw = cross_survey_blend(x_u8, idx, cap_idx, n_caps, device)
                xb = augment_views(xb)
                cb, pb = cell_idx[idx].to(device), pos[idx].to(device)
                logits, off_pred = model.training_outputs(xb)
                soft_t = make_soft_targets(pb, meta, model.n_cells, gw, gh)
                off_t = residual_offset_target(model, logits, pb, meta)
                loss = loss_fn(logits, off_pred, soft_t, off_t)
                opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    loss_sum += loss.item() * len(idx)
                    # top1 is measured on AUGMENTED views, so it reads lower
                    # than experiment 2's 0.79 by design; judge by eval.
                    # top1 is vs. the CONTAINING cell, so it now reads lower
                    # than exp 4's 0.83 BY DESIGN: border views are supposed
                    # to split mass across the tent's cells. Judge
                    # convergence by loss and by eval, not by this.
                    hit = (logits.argmax(dim=1) == cb).cpu().long()
                    correct += int(hit.sum())
                    mixed_c = mixed.cpu()
                    unmixed_c = ~mixed_c
                    nm_b = int(mixed_c.sum())
                    n_mixed += nm_b
                    correct_mixed += int(hit[mixed_c].sum())
                    correct_unmixed += int(hit[unmixed_c].sum())
                    if nm_b > 0:
                        mmw_sum += mmw; mmw_batches += 1
                    # Per-capture top1 over UNMIXED views only: a blended view
                    # has no single capture.
                    cb_caps = cap_idx[idx][unmixed_c]
                    cap_correct.scatter_add_(0, cb_caps, hit[unmixed_c])
                    cap_seen.scatter_add_(0, cb_caps, torch.ones_like(cb_caps))
                    conf_sum += float(F.softmax(logits, dim=1).max(dim=1).values.sum())
                    seen += len(idx)
            del x_u8, cell_idx, pos, cap_idx
        sched.step()
        final_top1 = correct / max(seen, 1)
        final_top1_by_cap = {
            nm: round(int(cap_correct[k]) / max(int(cap_seen[k]), 1), 4)
            for k, nm in enumerate(names)}
        blend_frac = n_mixed / max(seen, 1)
        top1_blend = correct_mixed / max(n_mixed, 1)
        top1_unmixed = correct_unmixed / max(seen - n_mixed, 1)
        mmw_epoch = mmw_sum / max(mmw_batches, 1)
        print(f"[{area}] epoch {epoch + 1}/{epochs} loss={loss_sum / max(seen, 1):.4f} "
              f"top1={final_top1:.4f} mean_max_softmax={conf_sum / max(seen, 1):.4f} "
              f"views={seen} lr={sched.get_last_lr()[0]:.2e} "
              f"top1_by_capture={final_top1_by_cap} "
              f"blend_frac={blend_frac:.4f} top1_blend={top1_blend:.4f} "
              f"top1_unmixed={top1_unmixed:.4f} mean_max_w={mmw_epoch:.4f}", flush=True)

    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = models_dir / f"{area}.onnx"
    export_onnx(model.cpu(), str(onnx_path))
    return {
        "area": area,
        "n_train_crops": epochs * len(crops) * len(imgs),
        "epochs": epochs,
        "device": device,
        "train_seconds": round(time.time() - t0, 1),
        "onnx_bytes": onnx_path.stat().st_size,
        "init": "from-scratch",  # §9: log init strategy per experiment
        "n_cells": model.n_cells,
        "distinct_vantages_seen": epochs * len(crops) * len(imgs),
        "full_lattice_epochs": True,
        "final_train_top1": round(final_top1, 4),
        "chunk_size": CHUNK_SIZE,
        "max_crops_per_bucket_arg": max_crops_per_bucket,
        "capture_sampling": ("paired-position interleave, shared heading per position, "
                             "cross-survey per-pixel blend (MIX) then per-view "
                             "photometric+sharpness nuisance randomisation"),
        "augmentation": AUG,
        "cross_survey_blend": MIX,
        "final_blend_fraction": round(blend_frac, 4),
        "final_top1_blend": round(top1_blend, 4),
        "final_top1_unmixed": round(top1_unmixed, 4),
        "final_blend_mean_max_weight": round(mmw_epoch, 4),
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
