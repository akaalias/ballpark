"""One REAL worked example per experiment for the gallery: take a fixed
held-out night crop, run it through the run's actual exported ONNX model,
and render (a) the crop the camera saw and (b) the full night map with the
model's actual internal probability field painted over it, its answer (×),
the true location (○), and the real miss distance.

The same crop is used for every experiment, so differences between two
experiments' figures ARE the mechanism change, not the example.

Presentation only (not frozen). Reads eval data strictly read-only; the
probability field is recovered from the exported ONNX graph itself (last
Softmax node), so the figure shows what the deployed artifact really
computes — no retraining, no reimplementation.
"""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from autoresearch.db import REPO_ROOT
from pipeline.common import area_dir, buckets, load_meta, px_to_lonlat
from pipeline.dataset import crop_center_norm, extract_crop, list_crops

AREA = "berlin"          # the only area on this branch (CLAUDE.md "BRANCH OVERRIDE")


def eval_bucket() -> str:
    """The EVAL bucket: the photograph the model is scored on (the held-out
    capture in the capture-holdout era; "asis" for a single-photograph
    area). Resolved lazily so importing this module needs no data on disk."""
    return next(iter(buckets(load_meta(AREA), "eval")))
MAP_PX = 640
ACCENT = (140, 47, 31)   # --accent #8c2f1f


def _haversine_m(lon1, lat1, lon2, lat2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _run_model(onnx_path: Path, frame: np.ndarray):
    """Run the exported model on one frame; also recover the internal
    probability field (output of the last Softmax node) when the graph has
    one — the same tensor the model decodes from, not a re-derivation."""
    import onnx
    import onnxruntime as ort

    x = frame.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    u, v, conf = (float(t) for t in sess.run(None, {"frame": x})[0][0])

    field = None
    try:
        m = onnx.load(str(onnx_path))
        softmaxes = [n for n in m.graph.node if n.op_type == "Softmax"]
        if softmaxes:
            name = softmaxes[-1].output[0]
            m.graph.output.append(
                onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
            sess2 = ort.InferenceSession(m.SerializeToString(),
                                         providers=["CPUExecutionProvider"])
            outs = [o.name for o in sess2.get_outputs()]
            res = sess2.run(None, {"frame": x})
            f = np.asarray(res[outs.index(name)], dtype=np.float64).reshape(-1)
            k = int(round(math.sqrt(f.size)))
            if k * k == f.size and k >= 4 and f.min() >= 0 and 0.5 < f.sum() < 1.5:
                field = f.reshape(k, k)  # [gy][gx], cell = gy*k+gx per contract
    except Exception:
        field = None
    return (u, v, conf), field


def _mark_truth(d: ImageDraw.ImageDraw, x, y):
    d.ellipse([x - 11, y - 11, x + 11, y + 11], outline=(255, 255, 255), width=3)
    d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 255, 255))


def _mark_pred(d: ImageDraw.ImageDraw, x, y):
    for dx, dy, w, col in ((0, 0, 7, (255, 255, 255)), (0, 0, 5, ACCENT)):
        d.line([x - 10, y - 10, x + 10, y + 10], fill=col, width=w - 2)
        d.line([x - 10, y + 10, x + 10, y - 10], fill=col, width=w - 2)


def generate(run_dir: Path) -> dict | None:
    """Build worked/{frame,map}.png + worked.json inside run_dir. Returns the
    JSON dict, or None when the run has no usable model for AREA."""
    onnx_path = run_dir / "models" / f"{AREA}.onnx"
    if not onnx_path.exists():
        return None
    out = run_dir / "worked"
    out.mkdir(exist_ok=True)

    meta = load_meta(AREA)
    crops = list_crops(AREA, meta["width"], meta["height"], "eval")
    c = crops[len(crops) // 2]  # deterministic: same eval crop for every run
    night = np.asarray(Image.open(area_dir(AREA) / "relight" / f"{eval_bucket()}.png"))
    frame = extract_crop(night, c["cx"], c["cy"], c["angle"])
    (u, v, conf), field = _run_model(onnx_path, frame)

    tu, tv = crop_center_norm(meta, c["cx"], c["cy"])
    w, h = meta["width"], meta["height"]
    lon_p, lat_p = px_to_lonlat(meta, u * w, v * h)
    lon_t, lat_t = px_to_lonlat(meta, tu * w, tv * h)
    miss_m = _haversine_m(lon_t, lat_t, lon_p, lat_p)

    Image.fromarray(frame).resize((256, 256), Image.LANCZOS).save(out / "frame.png")

    scale = MAP_PX / max(w, h)
    mw, mh = round(w * scale), round(h * scale)
    base = Image.open(area_dir(AREA) / "relight" / f"{eval_bucket()}.png") \
        .convert("RGB").resize((mw, mh), Image.LANCZOS)
    if field is not None:
        alpha = (field / field.max()) * 0.66
        a_img = Image.fromarray((alpha * 255).astype(np.uint8), "L") \
            .resize((mw, mh), Image.BILINEAR)
        overlay = Image.new("RGB", (mw, mh), ACCENT)
        base = Image.composite(overlay, base, a_img)
    d = ImageDraw.Draw(base, "RGBA")
    tx, ty = tu * w * scale, tv * h * scale
    px_, py_ = u * w * scale, v * h * scale
    d.line([tx, ty, px_, py_], fill=(255, 255, 255, 140), width=1)
    _mark_truth(d, tx, ty)
    _mark_pred(d, px_, py_)
    base.save(out / "map.png")

    info = {
        "area": AREA, "bucket": eval_bucket(), "miss_m": round(miss_m, 1),
        "conf": round(conf, 3),
        "has_field": field is not None,
        "field_k": None if field is None else int(field.shape[0]),
        "peak_pct": None if field is None else round(100 * float(field.max()), 3),
        "uniform_pct": None if field is None else round(100 / field.size, 3),
        # the actual coordinates, so the overview can state the contract
        # concretely: this frame -> these numbers
        "lat_true": round(lat_t, 6), "lon_true": round(lon_t, 6),
        "lat_pred": round(lat_p, 6), "lon_pred": round(lon_p, 6),
        "frame": "worked/frame.png", "map": "worked/map.png",
    }
    (out / "worked.json").write_text(json.dumps(info))
    return info


def ensure(artifacts_dir: str) -> dict | None:
    """Cached generate(): reuse worked.json if this run already has one."""
    run_dir = REPO_ROOT / (artifacts_dir or "")
    cached = run_dir / "worked" / "worked.json"
    if cached.exists():
        try:
            info = json.loads(cached.read_text())
            # re-render pre-coordinate caches so the overview can show
            # (lat, lon, confidence) rather than just a miss distance
            if "lat_true" in info:
                return info
        except (OSError, json.JSONDecodeError):
            pass
    try:
        return generate(run_dir)
    except Exception as e:  # never let a figure break the dashboard render
        print(f"worked example failed for {run_dir.name}: {e}")
        return None


if __name__ == "__main__":
    import sys
    print(generate(Path(sys.argv[1])))
