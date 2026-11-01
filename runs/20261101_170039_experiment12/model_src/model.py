"""AGENT-EDITABLE — model architecture + deployment inference wrapper.

The autoresearch loop may rewrite anything here (architecture, loss,
confidence mechanism, unified vs. dispatcher+specialists, pretrained init
per CLAUDE.md §3/§9) as long as:
  - train.py exports one ONNX per area at <out>/models/<area>.onnx taking a
    1x3x128x128 float32 input in [0,1] and returning [[u, v, conf]] with
    u, v normalized map coords and conf in [0,1];
  - the export passes pipeline/score.py's frozen deployment gates.

Current: localization as MAP-CELL CLASSIFICATION, from-scratch init, decoded by
LOCAL BELIEF POOLING. The map is cut into CELL_PX-square cells (Berlin:
55x54 = 2970); a conv trunk feeds a C-way softmax over cells plus a within-cell
offset head. The deployed decode pools belief over the 3x3 block of cells
centred on the argmax cell: confidence is the softmax mass on that whole
neighborhood, and the fix is the mass-weighted centroid of those cell centres,
nudged by the offset. Eval views stand 11-17 m off-lattice, so a view near a
cell border honestly splits its mass between adjacent cells — that is label
ambiguity, not place ambiguity, and pooling makes confidence measure "does the
model know WHERE it is" rather than "which arbitrary tile label applies".
Belief scattered across DISTANT cells still falls below threshold, so genuine
far-field ambiguity remains a safe abstention rather than a false fix. Cell
size is chosen so the parameterization alone suffices: a 128 m cell's
half-diagonal is 90.5 m, inside the 100 m usable radius, so naming the right
cell is already a usable fix. The TRAINING TARGETS are decode-consistent: the
cell target is a bilinear tent distribution over the <=4 nearest cell centres
(so a border view is taught the honest split its pooled decode will produce,
not an arbitrary one-cell label), and the offset target is the residual the
pooled centroid leaves rather than the position that centroid already encodes.

TWO-MEMBER COMMITTEE (capture-holdout era, experiment 7): the network is
N_MEMBERS = 2 fully independent Members (each its own trunk + lift + its own
cell head + its own offset head; no shared weights),
differing only by random initialisation and by their training augmentation
draws. Deployed decode: per-member softmax p_k; averaged belief
pbar = mean_k p_k; the argmax cell of pbar and the same 3x3 block mask around
it; conf = MIN over members of member k's own mass on that consensus block —
each member must independently put its belief there, so a frame on which the
two members name different blocks becomes an abstention. The fix is pbar's
mass-weighted block centroid plus the mean of the members' offsets. Member
disagreement is epistemic uncertainty, the kind a never-seen photograph
produces; a single model's softmax mass is not.

NORMALISED-PROTOTYPE COSINE HEAD (capture-holdout era, experiment 11): each
member's affine cell head (w_c . f + b_c) is replaced by GAP feature ->
BatchNorm1d -> L2-normalise -> cosine against an L2-normalised, bias-free
prototype matrix [n_cells x D]. Member.forward() returns those COSINES; the
deployed logits are COS_SCALE * cos (no margin). In training only, loss_fn
subtracts an additive margin COS_MARGIN from the target cells' cosine in
proportion to the tent target (CosFace-style; full margin on the dominant
cell, fractional on its tent neighbours). Why: an affine head over an
unnormalised feature can raise its margin by growing feature and weight norms,
so it may keep one direction per survey inside a wide cone for each tile. With
norms fixed, the only way to lower the loss is angular, so every survey's
picture of a tile must point at that tile's single prototype. The offset head
still reads the raw GAP feature.

PRETRAINED TRUNK (capture-holdout era, experiment 12): each member's
from-scratch conv trunk is replaced by torchvision's MobileNetV3-Small stem +
first eight inverted-residual blocks (features[0..8], IMAGENET1K_V1, BSD-3;
128x128 -> 48x8x8, the same 8x8 grid as before), loaded strict from the baked
file model/pretrained/mnv3s_features8.pt (verbatim torchvision tensors, so
nothing is downloaded at train or inference time), followed by a trainable
1x1 lift 48 -> EMBED_DIM + BatchNorm + ReLU. Each member builds its own
independent copy of the weights. ImageNet mean/std normalisation of the [0,1]
input is applied inside Member.embed via registered buffers, so it bakes into
the ONNX graph and the input contract is unchanged. train.py fine-tunes the
pretrained blocks at TRUNK_LR_MULT (0.1) times the head learning rate, so the
ImageNet invariances are adapted to Berlin rather than overwritten by it.

The whole decode (softmax -> argmax -> neighborhood mask -> pooled centroid ->
offset -> conf) lives INSIDE the exported graph, because the scorer runs the
ONNX directly. Contract unchanged: 1x3x128x128 float32 in [0,1] ->
[[u, v, conf]], u/v normalised map coords, conf in [0,1].
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small

CELL_PX = 128  # map cell edge in raster px (== metres at this GSD)
N_MEMBERS = 2  # independent committee members (no shared weights)
EMBED_DIM = 112   # lift output / prototype / offset-head width (as exp 11)
TRUNK_BLOCKS = 9  # mobilenet_v3_small.features[:9] = stem + 8 IR blocks -> 48x8x8 at 128 px
TRUNK_OUT_CH = 48
PRETRAINED_TRUNK_PATH = Path(__file__).parent / "pretrained" / "mnv3s_features8.pt"
TRUNK_LR_MULT = 0.1  # pretrained blocks train at this multiple of the head LR
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
COS_SCALE = 30.0   # fixed logit scale s: deployed logits = s * cos
COS_MARGIN = 0.25  # training-only additive cosine margin m on the tent target


def grid_dims(meta: dict) -> tuple[int, int]:
    """(gw, gh): map-cell grid covering the area's raster."""
    return (math.ceil(meta["width"] / CELL_PX), math.ceil(meta["height"] / CELL_PX))


def build_pretrained_trunk() -> nn.Sequential:
    """MobileNetV3-Small features[0..8] with ImageNet weights loaded STRICT
    from the baked file (keys prefixed 'features.'). Call once per member so
    each owns an independent copy."""
    trunk = mobilenet_v3_small(weights=None).features[:TRUNK_BLOCKS]
    sd = torch.load(PRETRAINED_TRUNK_PATH, map_location="cpu", weights_only=True)
    trunk.load_state_dict({k[len("features."):]: v for k, v in sd.items()}, strict=True)
    return trunk


class Member(nn.Module):
    """One independent localizer: pretrained MobileNetV3-Small trunk -> 1x1
    lift -> GAP -> (C-way cell COSINES against normalised prototypes,
    within-cell offset)."""

    def __init__(self, n_cells: int):
        super().__init__()
        self.trunk = build_pretrained_trunk()
        self.lift = nn.Sequential(nn.Conv2d(TRUNK_OUT_CH, EMBED_DIM, 1, bias=False),
                                  nn.BatchNorm2d(EMBED_DIM), nn.ReLU(inplace=True))
        self.register_buffer("in_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("in_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.embed_bn = nn.BatchNorm1d(EMBED_DIM)
        self.prototypes = nn.Parameter(torch.empty(n_cells, EMBED_DIM))
        nn.init.normal_(self.prototypes, std=0.01)
        self.off_head = nn.Linear(EMBED_DIM, 2)

    def trunk_parameters(self):
        """Pretrained trunk parameters (fine-tuned at TRUNK_LR_MULT x LR)."""
        return list(self.trunk.parameters())

    def head_parameters(self):
        """Everything not in self.trunk: lift, embed_bn, prototypes, off_head."""
        trunk_ids = {id(p) for p in self.trunk.parameters()}
        return [p for p in self.parameters() if id(p) not in trunk_ids]

    def embed(self, x):
        """(e, f): e = unit-length centred embedding [B, D]; f = raw GAP
        feature [B, D]. x is float [0,1]; ImageNet normalisation is in-graph."""
        f = self.lift(self.trunk((x - self.in_mean) / self.in_std)).mean(dim=(2, 3))
        e = F.normalize(self.embed_bn(f), dim=1)
        return e, f

    def forward(self, x):
        e, f = self.embed(x)
        cos = F.linear(e, F.normalize(self.prototypes, dim=1))  # [B, n_cells] in [-1, 1]
        # tanh*0.5 keeps offsets inside the cell, in (-0.5, 0.5) cell units.
        return cos, torch.tanh(self.off_head(f)) * 0.5


class CellLocNet(nn.Module):
    """Committee of N_MEMBERS independent Members.

    forward() is the DEPLOYED path and returns [B, 3] = (u, v, conf).
    training_outputs() exposes each member's raw (cosines, offset) for the loss;
    logits are COS_SCALE * cosines.
    """

    def __init__(self, meta: dict):
        super().__init__()
        gw, gh = grid_dims(meta)
        self.gw, self.gh = gw, gh
        self.n_cells = gw * gh
        width, height = meta["width"], meta["height"]

        # Sequential construction from the already-seeded torch RNG gives each
        # member a different init; deliberately not re-seeded between members.
        self.members = nn.ModuleList(Member(self.n_cells) for _ in range(N_MEMBERS))

        # Cell-centre lookup tables, in normalized map coords. Registered as
        # buffers so they bake into the ONNX graph as constants.
        idx = np.arange(self.n_cells)
        cell_u = ((idx % gw) + 0.5) * CELL_PX / width
        cell_v = ((idx // gw) + 0.5) * CELL_PX / height
        self.register_buffer("cell_u", torch.tensor(cell_u, dtype=torch.float32))
        self.register_buffer("cell_v", torch.tensor(cell_v, dtype=torch.float32))
        self.scale_u = CELL_PX / width
        self.scale_v = CELL_PX / height

    def block_mask(self, idx):
        """Float [B, n_cells] mask of the 3x3 block of cells centred on each
        argmax cell idx [B] (int64). 1.6 cell-pitches per axis (Chebyshev)
        selects exactly the 3x3 block with float-safety margin — the next ring
        sits at 2.0 pitches. Edge cells simply have fewer neighbours in range;
        nothing is counted twice."""
        win_u = self.cell_u.index_select(0, idx)
        win_v = self.cell_v.index_select(0, idx)
        du = (self.cell_u.unsqueeze(0) - win_u.unsqueeze(1)).abs()
        dv = (self.cell_v.unsqueeze(0) - win_v.unsqueeze(1)).abs()
        return ((du <= 1.6 * self.scale_u) & (dv <= 1.6 * self.scale_v)).float()

    def training_outputs(self, x):
        """List of N_MEMBERS (cosines, offset) tuples, all on the same x."""
        return [m(x) for m in self.members]

    def forward(self, x):
        # Committee decode: averaged belief names the consensus block; each
        # member's OWN mass on that block is measured and the smaller is the
        # confidence, so a fix is trusted only if every member independently
        # put its belief there. Fix = pbar's block centroid + mean offset.
        outs = self.training_outputs(x)
        ps = [F.softmax(COS_SCALE * c, dim=1) for c, _ in outs]  # no margin deployed
        pbar = torch.stack(ps, 0).mean(0)
        idx = pbar.argmax(dim=1)
        mask = self.block_mask(idx)
        masses = torch.stack([(p * mask).sum(dim=1) for p in ps], 0)  # [N_MEMBERS, B]
        conf = masses.min(dim=0).values  # min of softmax sub-sums: in [0,1]
        q = pbar * mask
        w = q / q.sum(dim=1, keepdim=True).clamp_min(1e-9)
        off = torch.stack([o for _, o in outs], 0).mean(0)
        u = (w * self.cell_u.unsqueeze(0)).sum(dim=1) + off[:, 0] * self.scale_u
        v = (w * self.cell_v.unsqueeze(0)).sum(dim=1) + off[:, 1] * self.scale_v
        return torch.stack([u.clamp(0.0, 1.0), v.clamp(0.0, 1.0), conf], dim=1)


def build_model(meta: dict) -> nn.Module:
    return CellLocNet(meta)


def loss_fn(cos: torch.Tensor, offset_pred: torch.Tensor,
            soft_target: torch.Tensor, offset_target: torch.Tensor) -> torch.Tensor:
    """Additive-margin soft cross-entropy against a bilinear tent distribution
    over cell centres + smooth-L1 on a residual offset target.

    cos is the member's [B, n_cells] cosine matrix (NOT logits). The training
    logits are COS_SCALE * (cos - COS_MARGIN * t_hat), where t_hat is the tent
    target rescaled so its dominant cell is 1.0 (tent neighbours fractional,
    the label-smoothing floor ~0): the margin is applied in proportion to the
    target and only in training. soft_target is a dense [B, n_cells]
    distribution (already includes label smoothing); offset_target is [B, 2]
    in (-0.5, 0.5) cell units, the residual the pooled-centroid decode leaves.
    No confidence loss term, as before.
    """
    t_hat = soft_target / soft_target.max(dim=1, keepdim=True).values.clamp_min(1e-9)
    logits = COS_SCALE * (cos - COS_MARGIN * t_hat)
    ce = -(soft_target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    return ce + 1.0 * F.smooth_l1_loss(offset_pred, offset_target)


def export_onnx(model: nn.Module, path: str):
    model.eval()
    dummy = torch.zeros(1, 3, 128, 128)
    torch.onnx.export(model, dummy, path, input_names=["frame"],
                      output_names=["uvc"], opset_version=17, dynamo=False)


def estimate_position(frame: np.ndarray, onnx_path: str, meta: dict):
    """Deployment-shaped inference: frame (128x128x3 uint8) -> (lat, lon, conf).

    On-device this maps to the P4 runtime; here it runs the same ONNX artifact.
    """
    import onnxruntime as ort
    from pipeline.common import px_to_lonlat
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = frame.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    u, v, conf = sess.run(None, {"frame": x})[0][0]
    lon, lat = px_to_lonlat(meta, float(u) * meta["width"], float(v) * meta["height"])
    return lat, lon, float(conf)
