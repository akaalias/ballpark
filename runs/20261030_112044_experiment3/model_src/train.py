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

CROSS-SURVEY CONSISTENCY: mini-batches are drawn as whole POSITIONS
(BATCH_SIZE // n_caps positions x every capture), so each batch holds the
same ground in all training photographs under one shared heading. On top of
the cell/offset loss, model.consistency_loss pulls each photograph's
normalised trunk feature toward the stop-gradient mean of the others' at the
same position, weighted by CONS_WEIGHT. The mean cross-survey cosine is
logged per epoch. With a single capture the term is zero.

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

from model.model import (CELL_PX, build_model, consistency_loss, export_onnx,
                         grid_dims, loss_fn)
from pipeline.common import DATA_DIR, area_dir, buckets, load_meta
from pipeline.dataset import extract_crop, list_crops

CHUNK_SIZE = 4096  # lattice positions extracted at a time (bounds memory)
BATCH_SIZE = 64
CONS_WEIGHT = 2.0  # weight of the cross-survey feature-consistency loss


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
    # Rows are emitted POSITION-MAJOR: row p*K + k is position p, capture k.
    # The position-grouped batches and consistency_loss rely on this ordering.
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
    pos_per_batch = max(1, BATCH_SIZE // n_caps)
    cap_offsets = torch.arange(n_caps)
    model = build_model(meta).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # A 12x-longer schedule wants a decaying LR, not the constant 1e-3.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                       eta_min=1e-4)
    print(f"[{area}] {len(crops)} lattice positions, {model.n_cells} cells, "
          f"device={device}")
    final_top1 = 0.0
    final_top1_by_cap = {nm: 0.0 for nm in names}
    cons_sum, cos_sum, seen = 0.0, 0.0, 0
    for epoch in range(epochs):
        model.train()
        loss_sum, correct, conf_sum, seen = 0.0, 0, 0.0, 0
        cons_sum, cos_sum = 0.0, 0.0
        cap_correct = torch.zeros(n_caps, dtype=torch.int64)
        cap_seen = torch.zeros(n_caps, dtype=torch.int64)
        # One full pass over the whole lattice x every capture, shuffled, fresh
        # rotations; each chunk holds its positions in ALL captures.
        order = rng.permutation(len(crops))
        for start in range(0, len(order), pos_per_chunk):
            x_u8, cell_idx, pos, cap_idx = extract_chunk(
                meta, imgs, crops, order[start:start + pos_per_chunk], rng)
            n = len(x_u8)
            n_pos = n // n_caps
            assert n == n_pos * n_caps
            # Position-grouped batches: whole positions x every capture, rows
            # kept position-major so consistency_loss can pair them.
            pos_perm = torch.randperm(n_pos)
            for i in range(0, n_pos, pos_per_batch):
                pidx = pos_perm[i:i + pos_per_batch]
                idx = (pidx.unsqueeze(1) * n_caps + cap_offsets).reshape(-1)
                xb = x_u8[idx].to(device).float() / 255.0
                cb, pb = cell_idx[idx].to(device), pos[idx].to(device)
                logits, off_pred, feat = model.training_outputs(xb)
                soft_t = make_soft_targets(pb, meta, model.n_cells, gw, gh)
                off_t = residual_offset_target(model, logits, pb, meta)
                cons, cos_mean = consistency_loss(feat, n_caps)
                loss = loss_fn(logits, off_pred, soft_t, off_t) + CONS_WEIGHT * cons
                opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    loss_sum += loss.item() * len(idx)
                    cons_sum += cons.item() * len(idx)
                    cos_sum += float(cos_mean) * len(idx)
                    # top1 is vs. the CONTAINING cell, so it now reads lower
                    # than exp 4's 0.83 BY DESIGN: border views are supposed
                    # to split mass across the tent's cells. Judge
                    # convergence by loss and by eval, not by this.
                    hit = (logits.argmax(dim=1) == cb).cpu().long()
                    correct += int(hit.sum())
                    cb_caps = cap_idx[idx]
                    cap_correct.scatter_add_(0, cb_caps, hit)
                    cap_seen.scatter_add_(0, cb_caps, torch.ones_like(cb_caps))
                    conf_sum += float(F.softmax(logits, dim=1).max(dim=1).values.sum())
                    seen += len(idx)
            del x_u8, cell_idx, pos, cap_idx
        sched.step()
        final_top1 = correct / max(seen, 1)
        final_top1_by_cap = {
            nm: round(int(cap_correct[k]) / max(int(cap_seen[k]), 1), 4)
            for k, nm in enumerate(names)}
        print(f"[{area}] epoch {epoch + 1}/{epochs} loss={loss_sum / max(seen, 1):.4f} "
              f"top1={final_top1:.4f} mean_max_softmax={conf_sum / max(seen, 1):.4f} "
              f"cons={cons_sum / max(seen, 1):.4f} "
              f"xsurvey_cos={cos_sum / max(seen, 1):.4f} "
              f"views={seen} lr={sched.get_last_lr()[0]:.2e} "
              f"top1_by_capture={final_top1_by_cap}", flush=True)

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
        "capture_sampling": "paired-position interleave, shared heading per position, "
                            "position-grouped batches + cross-survey feature consistency",
        "captures": names,
        "positions_per_chunk": pos_per_chunk,
        "final_train_top1_by_capture": final_top1_by_cap,
        "consistency_weight": CONS_WEIGHT,
        "final_consistency_loss": round(cons_sum / max(seen, 1), 4),
        "final_xsurvey_cosine": round(cos_sum / max(seen, 1), 4),
        "positions_per_batch": pos_per_batch,
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
