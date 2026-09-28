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
    return meta, imgs, crops


def extract_chunk(meta, img, crops, indices, rng):
    """Extract one chunk of vantages; return (x_uint8, cell_idx, pos).

    `cell_idx` is the containing cell, kept only for the top1 log line — the
    loss targets are built from `pos`, the continuous raster-pixel position.
    """
    gw, _ = grid_dims(meta)
    xs, cells, pos = [], [], []
    for i in indices:
        c = crops[i]
        cx, cy = c["cx"], c["cy"]
        angle = float(rng.uniform(0, 360))  # heading augmentation
        xs.append(extract_crop(img, cx, cy, angle))
        cells.append((cy // CELL_PX) * gw + (cx // CELL_PX))
        pos.append((float(cx), float(cy)))
    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).contiguous()
    return (x, torch.tensor(cells, dtype=torch.int64),
            torch.tensor(pos, dtype=torch.float32))


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
    meta, imgs, crops = load_scene(area, data_dir)
    gw, gh = grid_dims(meta)
    model = build_model(meta).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # A 12x-longer schedule wants a decaying LR, not the constant 1e-3.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                       eta_min=1e-4)
    print(f"[{area}] {len(crops)} lattice positions, {model.n_cells} cells, "
          f"device={device}")
    final_top1 = 0.0
    for epoch in range(epochs):
        model.train()
        loss_sum, correct, conf_sum, seen = 0.0, 0, 0.0, 0
        for img in imgs:
            # One full pass over the whole lattice, shuffled, fresh rotations.
            order = rng.permutation(len(crops))
            for start in range(0, len(order), CHUNK_SIZE):
                x_u8, cell_idx, pos = extract_chunk(
                    meta, img, crops, order[start:start + CHUNK_SIZE], rng)
                n = len(x_u8)
                perm = torch.randperm(n)
                for i in range(0, n, BATCH_SIZE):
                    idx = perm[i:i + BATCH_SIZE]
                    xb = x_u8[idx].to(device).float() / 255.0
                    cb, pb = cell_idx[idx].to(device), pos[idx].to(device)
                    logits, off_pred = model.training_outputs(xb)
                    soft_t = make_soft_targets(pb, meta, model.n_cells, gw, gh)
                    off_t = residual_offset_target(model, logits, pb, meta)
                    loss = loss_fn(logits, off_pred, soft_t, off_t)
                    opt.zero_grad(); loss.backward(); opt.step()
                    with torch.no_grad():
                        loss_sum += loss.item() * len(idx)
                        # top1 is vs. the CONTAINING cell, so it now reads lower
                        # than exp 4's 0.83 BY DESIGN: border views are supposed
                        # to split mass across the tent's cells. Judge
                        # convergence by loss and by eval, not by this.
                        correct += int((logits.argmax(dim=1) == cb).sum())
                        conf_sum += float(F.softmax(logits, dim=1).max(dim=1).values.sum())
                        seen += len(idx)
                del x_u8, cell_idx, pos
        sched.step()
        final_top1 = correct / max(seen, 1)
        print(f"[{area}] epoch {epoch + 1}/{epochs} loss={loss_sum / max(seen, 1):.4f} "
              f"top1={final_top1:.4f} mean_max_softmax={conf_sum / max(seen, 1):.4f} "
              f"views={seen} lr={sched.get_last_lr()[0]:.2e}")

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
