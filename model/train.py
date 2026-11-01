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

COSINE HEAD (experiment 11): each member's forward() now returns COSINES
against L2-normalised cell prototypes, not logits. The scaled, margin-free
logits COS_SCALE * cos are what the residual offset target, top1, agreement
and the committee belief statistics read; the training-only additive margin
lives inside loss_fn, which takes the cosines. At the end of every epoch a
clean SURVEY DIAGNOSTIC (survey_diag) cuts 512 fixed training-lattice
positions at heading 0, no augmentation, from each TRAINING capture only, and
logs per member: the same-place vs random-pair cosine of the normalised
embedding across surveys, the within-place variance fraction (0 = one
embedding per place regardless of survey), the fraction of positions on which
the member names the same cell in every survey, and clean top1 per capture.

PRETRAINED TRUNK (experiment 12, deepened in experiment 13): each member's
trunk is ImageNet-pretrained MobileNetV3-Small features[0..10] (model.model,
BSD-3, loaded strict from the baked model/pretrained/mnv3s_features10.pt),
read at two depths (48x8x8 and 96x4x4 at 128 px; 14x14/7x7 since exp 15), each through its own trainable 1x1
lift to 56, concatenated to 112.
The optimiser has two parameter groups: the pretrained trunk blocks at
TRUNK_LR_MULT (0.1) x the head LR, everything else at 1e-3, both decayed by
one LambdaLR of the previous cosine shape (multiplier 1.0 -> 0.1 over the run:
head 1e-3 -> 1e-4 as before, trunk 1e-4 -> 1e-5). Trunk BatchNorm layers stay
in train mode so their running stats adapt to aerial imagery.

STYLE MIXING (experiment 14): model.model.MixStyle blends each training
view's per-channel feature mean/std after trunk blocks 1/3/6 with a batch
partner's (usually another survey's), p=0.5, Beta(0.1,0.1); it is identity in
eval mode, so survey_diag and the export measure the mixing-free embedding.

NATIVE-SCALE INPUT (experiment 15): crops are still cut and augmented at
CROP_PX = 128; the bilinear resize to INPUT_PX = 224 (MobileNetV3's ImageNet
resolution) lives inside model.model.Member.embed, so training views,
survey_diag and the ONNX export all see it identically. The two read levels
become 48x14x14 and 96x7x7.

Usage: python -m model.train --area berlin --out-dir runs/<id> [--epochs 2]
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model.model import (CELL_PX, COS_MARGIN, COS_SCALE, EMBED_DIM, INPUT_PX, LIFT_DIM,
                         MIXSTYLE_AFTER, MIXSTYLE_ALPHA, MIXSTYLE_P, N_MEMBERS, PRETRAINED_TRUNK_PATH, TRUNK_BLOCKS,
                         TRUNK_LR_MULT, TRUNK_MID_CH, TRUNK_OUT_CH, build_model,
                         export_onnx, grid_dims, loss_fn)
from pipeline.common import DATA_DIR, area_dir, buckets, load_meta
from pipeline.dataset import CROP_PX, WINDOW_PX, list_crops

MID_HW = INPUT_PX // 16   # mid read level spatial size (14 at 224 px)
DEEP_HW = INPUT_PX // 32  # deep read level spatial size (7 at 224 px)
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
    member_correct = torch.zeros(N_MEMBERS, dtype=torch.int64, device=device)
    agree = torch.zeros((), dtype=torch.int64, device=device)
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
            loss = torch.zeros((), device=device)
            member_logits = []
            for member in model.members:
                xb_k = augment_views(xb0)
                cos_k, off_k = member(xb_k)
                logits_k = COS_SCALE * cos_k  # deployed, margin-free logits
                off_t_k = residual_offset_target(model, logits_k, pb, meta)
                loss = loss + loss_fn(cos_k, off_k, soft_t, off_t_k)
                member_logits.append(logits_k)
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
            "agree": int(agree)}


def survey_diag(model, scene, crop_xy, meta, device, n_caps, diag_idx):
    """Clean cross-survey diagnostic on fixed TRAINING-lattice positions
    (diag_idx), heading 0, no augmentation, TRAINING captures only. Per
    member: xs_cos_same / xs_cos_diff (mean over capture pairs of the cosine
    of the normalised embedding for the same place / a fixed random pairing),
    wpv (within-place variance fraction, exp 9 probe definition), xsurvey_agree
    (same argmax cell in every capture), clean_top1 (per capture)."""
    gw, _ = grid_dims(meta)
    was_training = model.training
    model.eval()
    xy = crop_xy[torch.as_tensor(diag_idx, dtype=torch.int64)]
    P = len(xy)
    cells = ((xy[:, 1] // CELL_PX) * gw + xy[:, 0] // CELL_PX).numpy()
    E = [[None] * n_caps for _ in range(N_MEMBERS)]   # [member][cap] -> [P, D]
    A = [[None] * n_caps for _ in range(N_MEMBERS)]   # [member][cap] -> [P]
    with torch.no_grad():
        for k in range(n_caps):
            es = [[] for _ in range(N_MEMBERS)]
            args = [[] for _ in range(N_MEMBERS)]
            for s in range(0, P, 64):
                cx = xy[s:s + 64, 0].to(device)
                cy = xy[s:s + 64, 1].to(device)
                b = len(cx)
                cap = torch.full((b,), k, dtype=torch.int64, device=device)
                x = gpu_crops(scene, cap, cx, cy, torch.zeros(b, device=device))
                for m, member in enumerate(model.members):
                    e, _ = member.embed(x)
                    cos = F.linear(e, F.normalize(member.prototypes, dim=1))
                    es[m].append(e.cpu())
                    args[m].append(cos.argmax(dim=1).cpu())
            for m in range(N_MEMBERS):
                E[m][k] = torch.cat(es[m]).numpy().astype(np.float64)
                A[m][k] = torch.cat(args[m]).numpy()
    if was_training:
        model.train()
    perm = np.random.default_rng(54321).permutation(P)
    pairs = [(a, b) for a in range(n_caps) for b in range(a + 1, n_caps)]
    out = {"xs_cos_same": [], "xs_cos_diff": [], "wpv": [],
           "xsurvey_agree": [], "clean_top1": []}
    for m in range(N_MEMBERS):
        if pairs:
            same = np.mean([np.mean(np.sum(E[m][a] * E[m][b], axis=1)) for a, b in pairs])
            diff = np.mean([np.mean(np.sum(E[m][a] * E[m][b][perm], axis=1)) for a, b in pairs])
        else:
            same = diff = float("nan")
        X = np.stack(E[m])                                   # [K, P, D]
        flat = X.reshape(-1, X.shape[-1])
        Z = (X - flat.mean(0)) / (flat.std(0) + 1e-6)
        w = float(((Z - Z.mean(axis=0, keepdims=True)) ** 2).mean())
        Am = np.stack(A[m])                                  # [K, P]
        agree = float(np.mean(np.all(Am == Am[0:1], axis=0)))
        top1 = [round(float(np.mean(Am[k] == cells)), 4) for k in range(n_caps)]
        out["xs_cos_same"].append(round(float(same), 4))
        out["xs_cos_diff"].append(round(float(diff), 4))
        out["wpv"].append(round(w, 4))
        out["xsurvey_agree"].append(round(agree, 4))
        out["clean_top1"].append(top1)
    return out


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
    print(f"[{area}] trunk: mobilenet_v3_small features[0..{TRUNK_BLOCKS - 1}] "
          f"IMAGENET1K_V1 loaded strict from {PRETRAINED_TRUNK_PATH.name} "
          f"x {N_MEMBERS} members, input {CROP_PX}->{INPUT_PX} px bilinear (in-graph), "
          f"lifts {TRUNK_MID_CH}->{LIFT_DIM} @{MID_HW}x{MID_HW} + "
          f"{TRUNK_OUT_CH}->{LIFT_DIM} @{DEEP_HW}x{DEEP_HW} -> {EMBED_DIM}, "
          f"trunk lr x{TRUNK_LR_MULT}, MixStyle after blocks {list(MIXSTYLE_AFTER)} "
          f"p={MIXSTYLE_P} alpha={MIXSTYLE_ALPHA} (train only)", flush=True)
    trunk_params = [p for m in model.members for p in m.trunk_parameters()]
    head_params = [p for m in model.members for p in m.head_parameters()]
    all_ids = [id(p) for p in model.parameters()]
    grouped = [id(p) for p in trunk_params + head_params]
    assert len(grouped) == len(all_ids) == len(set(grouped)) and set(grouped) == set(all_ids), \
        "optimiser groups must cover every parameter exactly once"
    opt = torch.optim.Adam([{"params": trunk_params, "lr": 1e-3 * TRUNK_LR_MULT},
                            {"params": head_params, "lr": 1e-3}])
    # Same cosine shape as before for both groups: multiplier 1.0 -> 0.1 over
    # the run (head 1e-3 -> 1e-4, trunk 1e-4 -> 1e-5), stepped once per epoch.
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * e / epochs)))
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
    # Fixed for the run: clean survey-diagnostic positions (training lattice).
    diag_idx = np.random.default_rng(12345).choice(
        len(crop_xy), min(512, len(crop_xy)), replace=False)
    diag = {}
    for epoch in range(epochs):
        stats = run_epoch(model, opt, meta, scene, crop_xy, rng, device,
                          n_caps, pos_per_chunk, gw, gh)
        sched.step()
        diag = survey_diag(model, scene, crop_xy, meta, device, n_caps, diag_idx)
        seen = stats["seen"]
        final_top1 = stats["correct"] / max(seen, 1)
        final_top1_by_cap = {
            nm: round(stats["cap_correct"][k] / max(stats["cap_seen"][k], 1), 4)
            for k, nm in enumerate(names)}
        final_member_top1 = [round(c / max(seen, 1), 4) for c in stats["member_correct"]]
        final_agree = round(stats["agree"] / max(seen, 1), 4)
        print(f"[{area}] epoch {epoch + 1}/{epochs} loss={stats['loss_sum'] / max(seen, 1):.4f} "
              f"top1={final_top1:.4f} mean_max_softmax={stats['conf_sum'] / max(seen, 1):.4f} "
              f"views={seen} lr={opt.param_groups[1]['lr']:.2e} "
              f"trunk_lr={opt.param_groups[0]['lr']:.2e} "
              f"top1_by_capture={final_top1_by_cap} "
              f"member_top1={final_member_top1} agree_rate={final_agree:.4f} "
              f"xs_cos_same={diag['xs_cos_same']} xs_cos_diff={diag['xs_cos_diff']} "
              f"wpv={diag['wpv']} xsurvey_agree={diag['xsurvey_agree']} "
              f"clean_top1={diag['clean_top1']}", flush=True)

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
        "init": ("pretrained:mobilenet_v3_small IMAGENET1K_V1 features[0..10] "
                 "(torchvision, BSD-3), fine-tuned at 0.1x LR"),  # §9
        "n_cells": model.n_cells,
        "distinct_vantages_seen": epochs * len(crops) * n_caps,
        "full_lattice_epochs": True,
        "final_train_top1": round(final_top1, 4),
        "chunk_size": CHUNK_SIZE,
        "max_crops_per_bucket_arg": max_crops_per_bucket,
        "capture_sampling": ("paired-position interleave, shared heading per position, "
                             "per-view photometric+sharpness nuisance randomisation, "
                             "two-member committee with independent augmentation draws per member, "
                             "normalised-prototype cosine head with additive margin, "
                             "ImageNet-pretrained MobileNetV3-Small features[0..10] trunk per member (0.1x LR), "
                             "two-level 8x8 + 4x4 place feature, "
                             "MixStyle cross-survey feature-statistic mixing after trunk "
                             "blocks 1/3/6 (train only), "
                             f"frame resized in-graph {CROP_PX}->{INPUT_PX} px before the trunk"),
        "input_px": INPUT_PX,
        "mixstyle": {"after_blocks": list(MIXSTYLE_AFTER), "p": MIXSTYLE_P,
                     "alpha": MIXSTYLE_ALPHA},
        "augmentation": AUG,
        "captures": names,
        "positions_per_chunk": pos_per_chunk,
        "final_train_top1_by_capture": final_top1_by_cap,
        "n_members": N_MEMBERS,
        "trunk": (f"mobilenet_v3_small features[0..10], two-level 1x1 lift "
                  f"{TRUNK_MID_CH}->{LIFT_DIM} ({MID_HW}x{MID_HW}) + "
                  f"{TRUNK_OUT_CH}->{LIFT_DIM} ({DEEP_HW}x{DEEP_HW}) "
                  f"-> {EMBED_DIM}"),
        "embed_dim": EMBED_DIM,
        "trunk_lr_mult": TRUNK_LR_MULT,
        "pretrained_file": PRETRAINED_TRUNK_PATH.name,
        "final_member_top1": final_member_top1,
        "final_member_agree_rate": final_agree,
        "head": (f"normalised-prototype cosine, BatchNorm1d embedding, s={COS_SCALE}, "
                 f"train-only additive margin m={COS_MARGIN} on the tent target"),
        "cos_scale": COS_SCALE,
        "cos_margin": COS_MARGIN,
        "final_xs_cos_same": diag.get("xs_cos_same"),
        "final_xs_cos_diff": diag.get("xs_cos_diff"),
        "final_wpv": diag.get("wpv"),
        "final_xsurvey_agree": diag.get("xsurvey_agree"),
        "final_clean_top1": diag.get("clean_top1"),
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
