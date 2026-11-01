"""Call the deployed Modal trainer for one area and write results to out-dir.

Produces the SAME layout as local model.train (models/<area>.onnx +
train_info.json), so loop.sh's merge/score steps are unchanged.

Usage: python infra/modal_client.py <area> <local-out-dir> <epochs> <max-crops-per-bucket>
"""
import pathlib
import sys
import time

import modal

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from infra.aws_train import package_data  # noqa: E402  (same guarded file list)
from pipeline.common import DATA_DIR, area_dir  # noqa: E402

VOLUME = "ballpark-train-data"


def ensure_data(area):
    """Upload meta.json + the training PNGs once per content hash. package_data
    refuses if an eval capture would be shipped; only its file list is used."""
    tar_path, digest = package_data(area)
    vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
    marker = f"{area}/.package-{digest}"
    try:
        names = {e.path for e in vol.listdir(area)}
    except Exception:
        names = set()
    if marker in names:
        return
    import tarfile
    with tarfile.open(tar_path) as t:
        members = t.getnames()
    root = area_dir(area, DATA_DIR).parent
    print(f"[modal] uploading {len(members)} training files to volume {VOLUME}...", flush=True)
    t0 = time.time()
    with vol.batch_upload(force=True) as b:
        for m in members:
            b.put_file(str(root / m), m)
        b.put_file(__file__, marker)   # any small file works as the marker
    print(f"[modal] uploaded in {time.time() - t0:.0f}s", flush=True)


def main():
    area, out_dir, epochs, crops = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    ensure_data(area)
    fn = modal.Function.from_name("lowlight-train", "train_area")
    t0 = time.time()
    res = fn.remote(area,
                    pathlib.Path("model/model.py").read_text(),
                    pathlib.Path("model/train.py").read_text(),
                    epochs, crops)
    if not res.get("ok"):
        sys.stderr.write("modal train failed on the GPU:\n" + (res.get("log") or "") + "\n")
        sys.exit(1)
    out = pathlib.Path(out_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "models" / f"{area}.onnx").write_bytes(res["onnx"])
    (out / "train_info.json").write_text(res["train_info"])
    print(f"modal train OK: {area} ({time.time() - t0:.0f}s call wall time)")


if __name__ == "__main__":
    main()
