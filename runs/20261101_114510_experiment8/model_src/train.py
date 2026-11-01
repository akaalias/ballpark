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

COMMITTEE (experiment 7): the model is two fully independent members
(model.model.N_MEMBERS) whose deployed confidence is the smaller of the two
members' belief on the consensus block. Each mini-batch is cut once, then every
member gets its OWN independent augment_views() draw of those crops and its own
loss (tent cross-entropy + residual offset target from that member's own
logits); the losses are summed into one optimiser step. There is no agreement
term: members differ by init and augmentation draws, and that diversity IS the
mechanism — disagreement between independently trained models is epistemic
uncertainty, the confidence gate that survives a change of photograph.
Per-member top1 and the member agreement rate are logged per epoch.

CROSS-SURVEY DINOv2 DISTILLATION (experiment 8): a frozen DINOv2 ViT-S/14
with registers (Apache-2.0, torch.hub facebookresearch/dinov2) is a
TRAINING-ONLY teacher. For every mini-batch the SAME position and heading is
cut from a DIFFERENT training capture (the partner survey), clean, and fed to
the teacher at 112 px, whose 8x8 patch-token grid is the target. Each
member's 8x8 feature map of its AUGMENTED own-survey crop goes through a
train-only 1x1-conv projector and is pulled toward that target by
(1 - cosine) per position, weighted DISTILL_WEIGHT. The partner is a different
survey so canopy, shadows, cars and colour pipeline — which differ between
surveys — cannot help the student match the target; only shared geometry can.
The teacher tokens are batch-centred before L2 normalisation so the common
token component cannot make the cosine vacuous (experiment 3's lesson). The
teacher and the projectors are discarded at export (the deployed graph is
unchanged); no teacher weight file is kept in the repo — torch.hub's own
cache is used. Per-member distillation loss and student-teacher cosine are
logged per epoch.

Usage: python -m model.train --area berlin --out-dir runs/<id> [--epochs 2]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from model.model import (CELL_PX, MEMBER_CHANS, N_MEMBERS, build_model,
                         export_onnx, grid_dims, loss_fn)
from pipeline.common import DATA_DIR, area_dir, buckets, load_meta
from pipeline.dataset import CROP_PX, WINDOW_PX, list_crops

CHUNK_SIZE = 4096  # lattice positions extracted at a time (bounds memory)
BATCH_SIZE = 64
# Per-view photometric + sharpness nuisance ranges (training only).
AUG = dict(blur_p=0.7, blur_sigma_max=1.5, blur_taps=7, exposure=(0.75, 1.30),
           channel_gain=(0.85, 1.15), contrast=(0.6, 1.4),
           brightness=(-0.12, 0.12), gamma=(0.7, 1.4), saturation=(0.5, 1.5),
           noise_sigma_max=0.03)
# Training-only cross-survey distillation teacher (never exported).
TEACHER_HUB = ("facebookresearch/dinov2", "dinov2_vits14_reg")
TEACHER_RES = 112        # 112/14 = 8 patch tokens per side == the student's 8x8 map
TEACHER_DIM = 384
PROJ_WIDTH = 512
DISTILL_WEIGHT = 5.0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_teacher(device, area=""):
    """Frozen DINOv2 ViT-S/14-reg from torch.hub's own cache (downloaded on
    first use). Training only; never exported."""
    t = torch.hub.load(*TEACHER_HUB, trust_repo=True)
    t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    n = sum(p.numel() for p in t.parameters()) / 1e6
    print(f"[{area}] teacher dinov2_vits14_reg loaded, {n:.1f}M params, frozen, "
          f"res {TEACHER_RES}", flush=True)
    return t.to(device)


def teacher_targets(teacher, x_clean, mean, std):
    """x_clean: float [B,3,128,128] in [0,1] on the device (UNaugmented).
    Returns [B, TEACHER_DIM, 8, 8]: batch-centred, L2-normalised per position
    patch tokens."""
    B = x_clean.shape[0]
    y = F.interpolate(x_clean, size=(TEACHER_RES, TEACHER_RES), mode="bilinear",
                      antialias=True, align_corners=False)
    y = (y - mean) / std
    g = TEACHER_RES // 14
    with torch.no_grad():
        if y.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                tok = teacher.forward_features(y)["x_norm_patchtokens"]
        else:
            tok = teacher.forward_features(y)["x_norm_patchtokens"]
        tok = tok.float()                                          # [B, 64, 384]
        t = tok.transpose(1, 2).reshape(B, TEACHER_DIM, g, g)
        # Batch-centre: remove the common token component so the cosine is
        # not vacuous (experiment 3's lesson).
        t = t - t.mean(dim=(0, 2, 3), keepdim=True)
        return F.normalize(t, dim=1)


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

    Mirrors forward()'s pooling exactly, using the model's own block_mask()
    and buffers so the grid geometry stays single-sourced. Called once per
    committee member with that member's own logits. Early in training the
    argmax is noisy, so this target is nonstationary; that is expected and
    bounded by the clamp.
    """
    with torch.no_grad():
        p = F.softmax(logits, dim=1)
        idx = p.argmax(dim=1)
        mask = model.block_mask(idx)
        q = p * mask
        m = q.sum(dim=1).clamp_min(1e-9)
        cu = (q * model.cell_u.unsqueeze(0)).sum(dim=1) / m
        cv = (q * model.cell_v.unsqueeze(0)).sum(dim=1) / m
        true_u = pos[:, 0] / meta["width"]; true_v = pos[:, 1] / meta["height"]
        ru = ((true_u - cu) / model.scale_u).clamp(-0.5, 0.5)
        rv = ((true_v - cv) / model.scale_v).clamp(-0.5, 0.5)
        return torch.stack([ru, rv], dim=1)


def run_epoch(model, opt, meta, scene, crop_xy, rng, device, n_caps,
              pos_per_chunk, gw, gh, projectors, teacher, t_mean, t_std,
              max_chunks=None):
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
    member_correct = torch.zeros(N_MEMBERS, dtype=torch.int64, device=device)
    agree = torch.zeros((), dtype=torch.int64, device=device)
    distill_sum = torch.zeros(N_MEMBERS, device=device)
    cos_sum = torch.zeros(N_MEMBERS, device=device)
    projectors.train()
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
            # Cut once; each member gets its OWN independent augmentation draw.
            xb0 = gpu_crops(scene, cap_b, cx[idx], cy[idx], ang[idx])
            cb, pb = cell_idx[idx], pos[idx]
            soft_t = make_soft_targets(pb, meta, model.n_cells, gw, gh)
            # Cross-survey teacher target: SAME position and heading, cut
            # clean from a DIFFERENT training capture (single-capture areas
            # distil from the same crop). Shared by both members.
            if n_caps > 1:
                cap_p = (cap_b + torch.randint(1, n_caps, (B,), device=device)) % n_caps
            else:
                cap_p = cap_b
            xb_p = gpu_crops(scene, cap_p, cx[idx], cy[idx], ang[idx])
            tgt = teacher_targets(teacher, xb_p, t_mean, t_std)
            loss = torch.zeros((), device=device)
            member_logits = []
            for k, member in enumerate(model.members):
                xb_k = augment_views(xb0)
                fmap_k = member.feature_map(xb_k)
                f_k = fmap_k.mean(dim=(2, 3))
                logits_k = member.cls_head(f_k)
                off_k = torch.tanh(member.off_head(f_k)) * 0.5  # == Member.forward
                z_k = F.normalize(projectors[k](fmap_k), dim=1)
                cos_k = (z_k * tgt).sum(dim=1)                  # [B,8,8]
                distill_k = (1.0 - cos_k).mean()
                off_t_k = residual_offset_target(model, logits_k, pb, meta)
                loss = loss + loss_fn(logits_k, off_k, soft_t, off_t_k) \
                    + DISTILL_WEIGHT * distill_k
                member_logits.append(logits_k)
                distill_sum[k] += distill_k.detach() * B
                cos_sum[k] += cos_k.detach().mean() * B
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                loss_sum += loss.detach() * B
                member_args = [l.argmax(dim=1) for l in member_logits]
                for k, a in enumerate(member_args):
                    member_correct[k] += (a == cb).long().sum()
                agree += (member_args[0] == member_args[1]).long().sum()
                # Committee belief: mean of the member softmaxes.
                pbar = torch.stack([F.softmax(l, dim=1) for l in member_logits], 0).mean(0)
                # top1 is measured on AUGMENTED views, so it reads lower
                # than experiment 2's 0.79 by design; judge by eval.
                # top1 is vs. the CONTAINING cell, so it now reads lower
                # than exp 4's 0.83 BY DESIGN: border views are supposed
                # to split mass across the tent's cells. Judge
                # convergence by loss and by eval, not by this.
                hit = (pbar.argmax(dim=1) == cb).long()
                correct += hit.sum()
                cap_correct.scatter_add_(0, cap_b, hit)
                cap_seen.scatter_add_(0, cap_b, torch.ones_like(cap_b))
                conf_sum += pbar.max(dim=1).values.sum()
                seen += B
    return {"loss_sum": float(loss_sum), "correct": int(correct),
            "conf_sum": float(conf_sum), "seen": seen,
            "cap_correct": cap_correct.cpu().tolist(),
            "cap_seen": cap_seen.cpu().tolist(),
            "member_correct": member_correct.cpu().tolist(),
            "agree": int(agree),
            "distill": distill_sum.cpu().tolist(),
            "cos": cos_sum.cpu().tolist()}


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
    # Training-only projectors (one per member), discarded at export.
    projectors = nn.ModuleList(
        nn.Sequential(nn.Conv2d(MEMBER_CHANS[-1], PROJ_WIDTH, 1), nn.ReLU(inplace=True),
                      nn.Conv2d(PROJ_WIDTH, TEACHER_DIM, 1))
        for _ in range(N_MEMBERS)).to(device)
    teacher = load_teacher(device, area)
    t_mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    t_std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    opt = torch.optim.Adam(list(model.parameters()) + list(projectors.parameters()),
                           lr=1e-3)
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
    final_member_top1 = [0.0] * N_MEMBERS
    final_agree = 0.0
    final_distill = [0.0] * N_MEMBERS
    final_cos = [0.0] * N_MEMBERS
    for epoch in range(epochs):
        stats = run_epoch(model, opt, meta, scene, crop_xy, rng, device,
                          n_caps, pos_per_chunk, gw, gh, projectors, teacher,
                          t_mean, t_std)
        sched.step()
        seen = stats["seen"]
        final_top1 = stats["correct"] / max(seen, 1)
        final_top1_by_cap = {
            nm: round(stats["cap_correct"][k] / max(stats["cap_seen"][k], 1), 4)
            for k, nm in enumerate(names)}
        final_member_top1 = [round(c / max(seen, 1), 4) for c in stats["member_correct"]]
        final_agree = round(stats["agree"] / max(seen, 1), 4)
        final_distill = [round(d / max(seen, 1), 4) for d in stats["distill"]]
        final_cos = [round(c / max(seen, 1), 4) for c in stats["cos"]]
        print(f"[{area}] epoch {epoch + 1}/{epochs} loss={stats['loss_sum'] / max(seen, 1):.4f} "
              f"top1={final_top1:.4f} mean_max_softmax={stats['conf_sum'] / max(seen, 1):.4f} "
              f"views={seen} lr={sched.get_last_lr()[0]:.2e} "
              f"top1_by_capture={final_top1_by_cap} "
              f"member_top1={final_member_top1} agree_rate={final_agree:.4f} "
              f"distill={final_distill} student_teacher_cos={final_cos}", flush=True)

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
        # §9: log init strategy per experiment
        "init": "from-scratch (student); training-only teacher pretrained:dinov2_vits14_reg",
        "n_cells": model.n_cells,
        "distinct_vantages_seen": epochs * len(crops) * n_caps,
        "full_lattice_epochs": True,
        "final_train_top1": round(final_top1, 4),
        "chunk_size": CHUNK_SIZE,
        "max_crops_per_bucket_arg": max_crops_per_bucket,
        "capture_sampling": ("paired-position interleave, shared heading per position, "
                             "per-view photometric+sharpness nuisance randomisation, "
                             "two-member committee with independent augmentation draws per member, "
                             "cross-survey DINOv2 patch distillation"),
        "augmentation": AUG,
        "captures": names,
        "positions_per_chunk": pos_per_chunk,
        "final_train_top1_by_capture": final_top1_by_cap,
        "n_members": N_MEMBERS,
        "member_chans": MEMBER_CHANS,
        "final_member_top1": final_member_top1,
        "final_member_agree_rate": final_agree,
        "distillation": {
            "teacher": "dinov2_vits14_reg (torch.hub facebookresearch/dinov2, Apache-2.0)",
            "teacher_res": TEACHER_RES,
            "teacher_dim": TEACHER_DIM,
            "proj_width": PROJ_WIDTH,
            "weight": DISTILL_WEIGHT,
            "target": ("batch-centred L2-normalised patch tokens of the SAME position+heading "
                       "in a DIFFERENT training capture (clean crop)"),
            "student_input": "augmented own-survey crop",
        },
        "final_distill_loss": final_distill,
        "final_student_teacher_cos": final_cos,
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
