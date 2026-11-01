"""Train ONE area on a throwaway AWS GPU machine and bring the result home.

    .venv/bin/python infra/aws_train.py <area> <local-out-dir> <epochs> <max-crops-per-bucket>

Same contract as infra/remote_train.sh and infra/modal_client.py: on success
<local-out-dir> holds models/<area>.onnx + train_info.json and the exit code
is 0; anything else is a failed train (loop.sh scores it 1e9). Issue #3.

How it works
- The laptop stays the only git writer. The machine gets a tarball of
  model/ + pipeline/ + areas.yaml and NO git, NO AWS credentials, NO SSH:
  it downloads and uploads only through presigned S3 links baked into its
  boot script, and it has no open ports.
- Data: meta.json + the role: train relight PNGs only, packed once and cached
  in S3 by content hash. The eval capture is never packed — package_data()
  refuses if any eval bucket name appears in what it is about to ship.
- Machine: a one-time spot instance (default g6.xlarge, L4), cheapest
  availability zone first, falling back to on-demand if no spot capacity.
- It cannot be forgotten: it shuts itself down (= terminates) when training
  ends, its boot script schedules a hard shutdown at AWS_MAX_MINUTES no
  matter what, and this script terminates it on every exit path.

Env (all optional): AWS_INSTANCE_TYPE (g6.xlarge), AWS_MAX_MINUTES (150),
AWS_CODE_REF (pack code from this git commit instead of the working tree),
AWS_ONDEMAND_FALLBACK (1), AWS_RETRIES (1, re-launch after a spot
interruption), AWS_AMI (override the Deep Learning AMI lookup).
Credentials come from .env (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
AWS_DEFAULT_REGION, default eu-central-1).
"""

import hashlib
import io
import os
import re
import sys
import tarfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import boto3  # noqa: E402
from botocore.config import Config  # noqa: E402
from botocore.exceptions import ClientError  # noqa: E402

from pipeline.common import DATA_DIR, area_dir, buckets, load_meta  # noqa: E402

CACHE = Path.home() / ".cache" / "ballpark-aws"
ONDEMAND_USD_H = {"g6.xlarge": 1.006, "g4dn.xlarge": 0.658, "g5.xlarge": 1.258,
                  "g6e.xlarge": 2.327}   # eu-central-1 list prices, 2026-10-29
CAPACITY_ERRORS = ("InsufficientInstanceCapacity", "SpotMaxPriceTooLow",
                   "MaxSpotInstanceCountExceeded", "Unsupported",
                   "InsufficientCapacity", "capacity-not-available")


def log(msg):
    print(f"[aws] {msg}", flush=True)


def load_env():
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    os.environ.setdefault("AWS_DEFAULT_REGION", "eu-central-1")


# ---------------------------------------------------------------- packaging

def package_data(area):
    """Tar of <area>/meta.json + relight/<train bucket>.png, cached by
    content hash. Refuses if anything eval-related would be shipped."""
    meta = load_meta(area, DATA_DIR)
    train = list(buckets(meta, "train"))
    evals = list(buckets(meta, "eval"))
    root = area_dir(area, DATA_DIR)
    files = [root / "meta.json"] + [root / "relight" / f"{b}.png" for b in train]
    for f in files:
        if any(e in f.name for e in evals):
            raise SystemExit(f"refusing to ship eval capture: {f}")
        if not f.exists():
            raise SystemExit(f"missing training file: {f}")
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        with open(f, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                h.update(block)
    digest = h.hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)
    tar_path = CACHE / f"data-{area}-{digest}.tar"
    if not tar_path.exists():
        tmp = tar_path.with_suffix(".tmp")
        with tarfile.open(tmp, "w") as t:          # PNGs are already compressed
            for f in files:
                t.add(f, arcname=str(f.relative_to(root.parent)))
        tmp.rename(tar_path)
    with tarfile.open(tar_path) as t:              # belt and braces
        names = t.getnames()
    leaked = [n for n in names if any(e in n for e in evals)]
    if leaked:
        raise SystemExit(f"refusing to ship eval capture: {leaked}")
    log(f"data package {tar_path.name}: {len(names)} files "
        f"({tar_path.stat().st_size / 1e6:.0f} MB), train captures {train}, "
        f"eval {evals} excluded")
    return tar_path, digest


def package_code():
    """model/ + pipeline/ + areas.yaml from the working tree, or from git
    commit $AWS_CODE_REF when set (benchmarks while the loop is editing
    model/)."""
    ref = os.environ.get("AWS_CODE_REF")
    if ref:
        import subprocess
        out = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar.gz", ref,
                              "model", "pipeline", "areas.yaml"],
                             capture_output=True, check=True).stdout
        log(f"code from git {ref}")
        return out
    buf = io.BytesIO()
    skip = lambda ti: None if ("__pycache__" in ti.name or ti.name.endswith(".pyc")) else ti
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for p in ("model", "pipeline", "areas.yaml"):
            t.add(REPO / p, arcname=p, filter=skip)
    return buf.getvalue()


# ---------------------------------------------------------------- S3

def ensure_bucket(s3, name, region):
    try:
        s3.head_bucket(Bucket=name)
        return
    except ClientError:
        pass
    log(f"creating bucket {name}")
    s3.create_bucket(Bucket=name,
                     CreateBucketConfiguration={"LocationConstraint": region})
    s3.put_public_access_block(Bucket=name, PublicAccessBlockConfiguration={
        "BlockPublicAcls": True, "IgnorePublicAcls": True,
        "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
    s3.put_bucket_lifecycle_configuration(Bucket=name, LifecycleConfiguration={
        "Rules": [{"ID": "expire-jobs", "Status": "Enabled",
                   "Filter": {"Prefix": "jobs/"},
                   "Expiration": {"Days": 7}}]})


def exists(s3, bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def preflight(urls):
    """Try every presigned link from the laptop, with curl exactly as the box
    uses it (no redirect following), before paying for a machine."""
    import subprocess
    def code(*args):
        return subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                               "-m", "30", *args], capture_output=True, text=True).stdout
    checks = {"code": code("-r", "0-0", urls["code"]),
              "data": code("-r", "0-0", urls["data"]),
              "log": code("-X", "PUT", "--data-binary", "", urls["log"])}   # empty: the box overwrites it
    bad = {k: v for k, v in checks.items() if v not in ("200", "206")}
    if bad:
        raise SystemExit(f"preflight failed, not launching: {bad}")
    log("preflight: download and upload links work")


# ---------------------------------------------------------------- EC2

def find_ami(ec2):
    if os.environ.get("AWS_AMI"):
        return os.environ["AWS_AMI"], "(AWS_AMI override)"
    imgs = ec2.describe_images(Owners=["amazon"], Filters=[
        {"Name": "name", "Values": ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch *Ubuntu*"]},
        {"Name": "architecture", "Values": ["x86_64"]},
        {"Name": "state", "Values": ["available"]}])["Images"]
    if not imgs:
        raise SystemExit("no Deep Learning PyTorch AMI found; set AWS_AMI")
    # Highest PyTorch version, then newest build: AWS keeps rebuilding old
    # versions (2.10-2.12 carry unpatched advisories), so newest-by-date alone
    # can pick an old torch.
    def key(i):
        m = re.search(r"PyTorch (\d+)\.(\d+)", i["Name"])
        return (tuple(map(int, m.groups())) if m else (0, 0), i["CreationDate"])
    best = max(imgs, key=key)
    return best["ImageId"], best["Name"]


def boot_script(area, epochs, crops, urls, max_minutes):
    return f"""#!/bin/bash
# Hard deadline first: whatever happens below, this machine is gone by then.
shutdown -h +{max_minutes}
exec > /var/log/ballpark.log 2>&1
LOG_URL='{urls["log"]}'
push_log() {{ curl -s -m 60 -X PUT --upload-file /var/log/ballpark.log "$LOG_URL" -o /dev/null; }}
finish() {{
  echo "$1" > /opt/job/out/exit_code
  tar czf /opt/job/result.tgz -C /opt/job/out .
  curl -fsS --retry 5 -m 300 -X PUT --upload-file /opt/job/result.tgz '{urls["result"]}' \\
    && echo "[box] result uploaded" || echo "[box] result upload FAILED"
  push_log; sync; shutdown -h now; exit 0
}}
( while sleep 30; do push_log; done ) &
echo "[box] boot $(date -u +%FT%TZ)"
mkdir -p /opt/job/out && cd /opt/job
curl -fsS --retry 5 -o code.tgz '{urls["code"]}' && tar xzf code.tgz || {{ echo "[box] code download failed"; finish 90; }}
curl -fsS --retry 5 -o data.tar '{urls["data"]}' && mkdir -p data && tar xf data.tar -C data && rm data.tar \\
  || {{ echo "[box] data download failed"; finish 91; }}
echo "[box] downloads done $(date -u +%FT%TZ)"
source /opt/pytorch/bin/activate || {{ echo "[box] no /opt/pytorch venv"; finish 92; }}
pip install -q onnx pyyaml pillow 2>&1 | tail -3
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c 'import torch; print("[box] torch", torch.__version__, "cuda", torch.cuda.is_available())'
T0=$(date +%s)
python -u -m model.train --area {area} --out-dir out --epochs {epochs} --max-crops-per-bucket {crops}
RC=$?
echo "[box] train exit=$RC seconds=$(( $(date +%s) - T0 ))"
finish $RC
"""


def launch(ec2, itype, ami, user_data, tag):
    """Spot in the cheapest AZ first, then the other AZs, then on-demand."""
    subnets = {s["AvailabilityZone"]: s["SubnetId"] for s in ec2.describe_subnets(
        Filters=[{"Name": "default-for-az", "Values": ["true"]}])["Subnets"]}
    if not subnets:
        raise SystemExit("no default subnets in this region")
    prices = {}
    for p in ec2.describe_spot_price_history(
            InstanceTypes=[itype], ProductDescriptions=["Linux/UNIX"],
            StartTime=datetime.now(timezone.utc) - timedelta(hours=1))["SpotPriceHistory"]:
        prices.setdefault(p["AvailabilityZone"], float(p["SpotPrice"]))
    zones = sorted(subnets, key=lambda z: prices.get(z, 99.0))
    attempts = [("spot", z) for z in zones]
    if os.environ.get("AWS_ONDEMAND_FALLBACK", "1") == "1":
        attempts += [("on-demand", z) for z in zones]
    common = dict(
        ImageId=ami, InstanceType=itype, MinCount=1, MaxCount=1,
        UserData=user_data, InstanceInitiatedShutdownBehavior="terminate",
        MetadataOptions={"HttpTokens": "required"},
        TagSpecifications=[{"ResourceType": "instance", "Tags": [
            {"Key": "Name", "Value": f"ballpark-{tag}"},
            {"Key": "Project", "Value": "ballpark"}]}])
    for market, zone in attempts:
        kw = dict(common, SubnetId=subnets[zone])
        if market == "spot":
            kw["InstanceMarketOptions"] = {"MarketType": "spot", "SpotOptions": {
                "SpotInstanceType": "one-time",
                "InstanceInterruptionBehavior": "terminate"}}
        try:
            inst = ec2.run_instances(**kw)["Instances"][0]
        except ClientError as e:
            code = e.response["Error"]["Code"]
            msg = e.response["Error"].get("Message", "")
            if any(c in code or c in msg for c in CAPACITY_ERRORS):
                log(f"{market} {zone}: no capacity ({code}), trying next")
                continue
            raise
        price = prices.get(zone) if market == "spot" else ONDEMAND_USD_H.get(itype)
        log(f"launched {inst['InstanceId']} {itype} {market} in {zone}"
            + (f" at ~${price:.3f}/h" if price else ""))
        return inst["InstanceId"], market, price
    raise SystemExit(f"no capacity for {itype} anywhere in the region")


# ---------------------------------------------------------------- one job

def run_job(sess, area, out_dir, epochs, crops, attempt):
    region = sess.region_name
    # Regional endpoint + SigV4: presigned links to a NEW bucket on the global
    # endpoint answer 307 for hours, and the box's curl does not follow them.
    s3 = sess.client("s3", region_name=region, config=Config(
        signature_version="s3v4", s3={"addressing_style": "virtual"}))
    ec2 = sess.client("ec2")
    account = sess.client("sts").get_caller_identity()["Account"]
    bucket = f"ballpark-train-{account}-{region}"
    ensure_bucket(s3, bucket, region)

    tar_path, digest = package_data(area)
    data_key = f"data/{area}-{digest}.tar"
    if not exists(s3, bucket, data_key):
        log(f"uploading data package once ({tar_path.stat().st_size / 1e6:.0f} MB)...")
        t = time.time()
        s3.upload_file(str(tar_path), bucket, data_key)
        log(f"uploaded in {time.time() - t:.0f}s")
    tag = f"{area}-{time.strftime('%Y%m%d-%H%M%S')}-{attempt}"
    job = f"jobs/{tag}"
    s3.put_object(Bucket=bucket, Key=f"{job}/code.tgz", Body=package_code())

    max_minutes = int(os.environ.get("AWS_MAX_MINUTES", "150"))
    ttl = max_minutes * 60 + 3600
    sign = lambda op, key: s3.generate_presigned_url(
        op, Params={"Bucket": bucket, "Key": key}, ExpiresIn=ttl)
    urls = {"code": sign("get_object", f"{job}/code.tgz"),
            "data": sign("get_object", data_key),
            "log": sign("put_object", f"{job}/box.log"),
            "result": sign("put_object", f"{job}/result.tgz")}
    preflight(urls)
    itype = os.environ.get("AWS_INSTANCE_TYPE", "g6.xlarge")
    ami, ami_name = find_ami(ec2)
    log(f"AMI {ami} {ami_name}")
    user_data = boot_script(area, epochs, crops, urls, max_minutes)

    iid, market, price = launch(ec2, itype, ami, user_data, tag)
    t_launch = time.time()
    shown = 0
    gone_since = None
    try:
        while True:
            time.sleep(20)
            try:   # stream the box's log into ours (the dashboard reads it)
                body = s3.get_object(Bucket=bucket, Key=f"{job}/box.log")["Body"].read()
                if len(body) > shown:
                    sys.stdout.write(body[shown:].decode(errors="replace"))
                    sys.stdout.flush()
                    shown = len(body)
            except ClientError:
                pass
            if exists(s3, bucket, f"{job}/result.tgz"):
                break
            try:
                state = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]
            except (ClientError, IndexError):
                continue          # a just-launched instance can be briefly unknown
            st = state["State"]["Name"]
            if st in ("shutting-down", "terminated", "stopped", "stopping"):
                gone_since = gone_since or time.time()
                if time.time() - gone_since > 90:   # grace for a late upload
                    reason = state.get("StateReason", {}).get("Message", "?")
                    log(f"machine {st} without a result: {reason}")
                    return "interrupted" if "Spot" in reason else "failed"
            if time.time() - t_launch > max_minutes * 60 + 600:
                log("past the hard deadline without a result")
                return "failed"
        buf = io.BytesIO(s3.get_object(Bucket=bucket, Key=f"{job}/result.tgz")["Body"].read())
        out_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=buf) as t:
            t.extractall(out_dir, filter="data")
        rc = (out_dir / "exit_code").read_text().strip()
        ok = rc == "0" and (out_dir / "models" / f"{area}.onnx").exists()
        return "ok" if ok else "failed"
    finally:
        try:
            ec2.terminate_instances(InstanceIds=[iid])
        except ClientError as e:
            log(f"WARNING terminate {iid} failed: {e}")
        mins = (time.time() - t_launch) / 60
        cost = f", ~${price * mins / 60:.2f}" if price else ""
        log(f"machine {iid} terminated after {mins:.1f} min ({market}{cost})")


def main():
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    area, out_dir, epochs, crops = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    load_env()
    sess = boto3.session.Session()
    retries = int(os.environ.get("AWS_RETRIES", "1"))
    for attempt in range(retries + 1):
        result = run_job(sess, area, out_dir, epochs, crops, attempt)
        if result == "ok":
            log("done")
            return 0
        if result != "interrupted":
            break
        log("spot interruption, launching again")
    log(f"FAILED ({result})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
