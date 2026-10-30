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

The whole decode (softmax -> argmax -> neighborhood mask -> pooled centroid ->
offset -> conf) lives INSIDE the exported graph, because the scorer runs the
ONNX directly.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CELL_PX = 128  # map cell edge in raster px (== metres at this GSD)


def grid_dims(meta: dict) -> tuple[int, int]:
    """(gw, gh): map-cell grid covering the area's raster."""
    return (math.ceil(meta["width"] / CELL_PX), math.ceil(meta["height"] / CELL_PX))


class CellLocNet(nn.Module):
    """Conv trunk -> GAP -> (C-way cell logits, within-cell offset).

    forward() is the DEPLOYED path and returns [B, 3] = (u, v, conf).
    training_outputs() exposes the raw (logits, offset) for the loss.
    """

    def __init__(self, meta: dict):
        super().__init__()
        gw, gh = grid_dims(meta)
        self.gw, self.gh = gw, gh
        self.n_cells = gw * gh
        width, height = meta["width"], meta["height"]

        chans = [3, 32, 64, 128, 160]
        layers = []
        for cin, cout in zip(chans, chans[1:]):
            layers += [nn.Conv2d(cin, cout, 3, stride=2, padding=1),
                       nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*layers)
        self.cls_head = nn.Linear(chans[-1], self.n_cells)
        self.off_head = nn.Linear(chans[-1], 2)

        # Cell-centre lookup tables, in normalized map coords. Registered as
        # buffers so they bake into the ONNX graph as constants.
        idx = np.arange(self.n_cells)
        cell_u = ((idx % gw) + 0.5) * CELL_PX / width
        cell_v = ((idx // gw) + 0.5) * CELL_PX / height
        self.register_buffer("cell_u", torch.tensor(cell_u, dtype=torch.float32))
        self.register_buffer("cell_v", torch.tensor(cell_v, dtype=torch.float32))
        self.scale_u = CELL_PX / width
        self.scale_v = CELL_PX / height

    def training_outputs(self, x):
        f = self.features(x).mean(dim=(2, 3))
        # tanh*0.5 keeps offsets inside the cell, in (-0.5, 0.5) cell units.
        return self.cls_head(f), torch.tanh(self.off_head(f)) * 0.5

    def forward(self, x):
        # Local belief pooling: confidence is the softmax mass on the 3x3 block
        # of cells centred on the argmax cell, and the fix is that block's
        # mass-weighted centroid (plus the within-cell offset). A border view
        # splitting mass between adjacent cells still names one place, so it
        # stays a fix; mass scattered across distant cells stays an abstention.
        logits, offset = self.training_outputs(x)
        p = F.softmax(logits, dim=1)
        idx = p.argmax(dim=1)
        win_u = self.cell_u.index_select(0, idx)
        win_v = self.cell_v.index_select(0, idx)
        # 1.6 cell-pitches per axis (Chebyshev) selects exactly the 3x3 block
        # with float-safety margin — the next ring sits at 2.0 pitches. Edge
        # cells simply have fewer neighbours in range; nothing is counted twice.
        du = (self.cell_u.unsqueeze(0) - win_u.unsqueeze(1)).abs()
        dv = (self.cell_v.unsqueeze(0) - win_v.unsqueeze(1)).abs()
        mask = ((du <= 1.6 * self.scale_u) & (dv <= 1.6 * self.scale_v)).float()
        q = p * mask
        conf = q.sum(dim=1)  # sub-sum of a softmax: in [0,1], >= single-cell max
        w = q / conf.unsqueeze(1).clamp_min(1e-9)
        u = (w * self.cell_u.unsqueeze(0)).sum(dim=1) + offset[:, 0] * self.scale_u
        v = (w * self.cell_v.unsqueeze(0)).sum(dim=1) + offset[:, 1] * self.scale_v
        return torch.stack([u.clamp(0.0, 1.0), v.clamp(0.0, 1.0), conf], dim=1)


def build_model(meta: dict) -> nn.Module:
    return CellLocNet(meta)


def loss_fn(logits: torch.Tensor, offset_pred: torch.Tensor,
            soft_target: torch.Tensor, offset_target: torch.Tensor) -> torch.Tensor:
    """Soft cross-entropy against a bilinear tent distribution over cell
    centres + smooth-L1 on a residual offset target. soft_target is a dense
    [B, n_cells] distribution (already includes label smoothing);
    offset_target is [B, 2] in (-0.5, 0.5) cell units, the residual the
    pooled-centroid decode leaves. No confidence loss term, as before.
    """
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
