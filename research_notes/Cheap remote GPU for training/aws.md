# AWS as a cheap remote GPU for short PyTorch training jobs (from a MacBook bash loop)

Research date: 2026-10-31. On-demand prices come from AWS's public price files published **2026-10-29** (pulled directly; see sources). Spot prices are harder to pin down: see the warning under Q1. Workload: one small CNN per experiment, 45 min to 4+ h on an M1 (MPS), ~1.1 GB of raster data that rarely changes, a few hundred KB of code that changes every run, and an output of ~3 MB ONNX plus a JSON. The laptop stays the only git writer.

## Q1. Which GPU instance types fit, and what do they cost (on-demand vs spot, eu-central-1 and us-east-1)?

### Takeaway
For a ~3 MB CNN, the cheapest sensible single-GPU choices are **g4dn.xlarge (T4 16 GB)** and **g6.xlarge (L4 24 GB)**. On-demand they cost **$0.658/h / $1.006/h in Frankfurt** and **$0.526/h / $0.805/h in us-east-1**. Spot is roughly 50–62% cheaper, which puts both at about **$0.21–0.41/h**. In Frankfurt, g6.xlarge has the best interruption rating of the cheap GPUs (<5%), while g4dn.xlarge is rated 15–20%. g5 (A10G) and g6e (L40S) cost more and bring nothing extra for a model this small.

### Cited Findings
**On-demand Linux $/h, AWS price file published 2026-10-29** — [AWS EC2 on-demand price JSON, Frankfurt](https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/EU%20(Frankfurt)/Linux/index.json); [same, N. Virginia](https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20(N.%20Virginia)/Linux/index.json)

| instance | GPU | vCPU / RAM | eu-central-1 OD | us-east-1 OD |
|---|---|---|---|---|
| g4dn.xlarge | 1× T4 16 GB | 4 / 16 GiB | $0.658 | $0.526 |
| g4dn.2xlarge | 1× T4 | 8 / 32 GiB | $0.940 | $0.752 |
| g5.xlarge | 1× A10G 24 GB | 4 / 16 GiB | $1.258 | $1.006 |
| g5.2xlarge | 1× A10G | 8 / 32 GiB | $1.516 | $1.212 |
| g6.xlarge | 1× L4 24 GB | 4 / 16 GiB | $1.006 | $0.805 |
| g6.2xlarge | 1× L4 | 8 / 32 GiB | $1.222 | $0.978 |
| g6e.xlarge | 1× L40S 48 GB | 4 / 32 GiB | $2.327 | $1.861 |
| g6f.large | fractional L4 | 2 / 8 GiB | $0.253 | $0.202 |
| g6f.xlarge | fractional L4 | 4 / 16 GiB | $0.297 | $0.238 |

- GPU and memory specs: g4dn.xlarge has a T4 with 16 GB, g5.xlarge an A10G with 24 GB, and g6.xlarge an L4 with 24 GB (compute capability 8.9, introduced 2024-04-04) and a 250 GB local NVMe — [Vantage g4dn](https://instances.vantage.sh/aws/ec2/g4dn.xlarge), [Vantage g5](https://instances.vantage.sh/aws/ec2/g5.xlarge), [Vantage g6](https://instances.vantage.sh/aws/ec2/g6.xlarge)
- **AWS Spot Instance Advisor** (data file last modified 2026-03-20). Savings vs on-demand and interruption-frequency band — [spot-advisor-data.json](https://spot-bid-advisor.s3.amazonaws.com/spot-advisor-data.json):

| instance | eu-central-1 savings / interruption | us-east-1 savings / interruption |
|---|---|---|
| g4dn.xlarge | 62% / 15–20% | 60% / 10–15% |
| g4dn.2xlarge | 64% / >20% | 61% / >20% |
| g5.xlarge | 50% / 15–20% | 56% / >20% |
| g6.xlarge | **60% / <5%** | 49% / >20% |
| g6.2xlarge | **63% / <5%** | 46% / 10–15% |
| g6e.xlarge | 43% / >20% | 50% / >20% |
| g6f.large | 73% / <5% | 67% / <5% |
| g6f.xlarge | 57% / 5–10% | 62% / 5–10% |

  The advisor's global average interruption rate is "<10%" — same file.
- **Implied spot prices** (on-demand × (1 − advisor savings)): g4dn.xlarge ≈ $0.25 Frankfurt / $0.21 us-east-1; g6.xlarge ≈ $0.40 Frankfurt / $0.41 us-east-1; g5.xlarge ≈ $0.63 / $0.44. This is arithmetic on the two sources above, not a quoted price.
- Vantage lists spot at **g4dn.xlarge $0.276, g5.xlarge $0.541, g6.xlarge $0.610**. The page is undated, and its region is unclear (its on-demand figures match us-east-1) — [Vantage g4dn](https://instances.vantage.sh/aws/ec2/g4dn.xlarge), [g5](https://instances.vantage.sh/aws/ec2/g5.xlarge), [g6](https://instances.vantage.sh/aws/ec2/g6.xlarge)
- **Stale-data warning.** AWS's public `spot.js` feed (`website.spot.ec2.aws.a2z.com/spot.js`) has a Last-Modified header of **13 May 2024**. It shows us-east-1 g6.xlarge spot at $0.105, about 13% of on-demand, which is not credible. Third-party aggregators that use this feed, such as [ec2.shop](https://ec2.shop), repeat the stale numbers: ec2.shop shows g4dn.xlarge spot at $0.2352 in Frankfurt and $0.2204 in us-east-1, g6.xlarge at $0.105 in us-east-1, and "NA" for g6 in Frankfurt. It also labels g6e as "NVIDIA L4 48GB", but g6e has an L40S. Do not quote these as current.
- Spot instances can be interrupted with a 2-minute warning. SageMaker's managed spot training needs checkpointing to resume after an interruption — [SageMaker Managed Spot Training docs](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html)

### Inferences
- **Recommended default: g6.xlarge spot in eu-central-1.** It costs about $0.40/h, has a <5% interruption band, uses a newer GPU architecture (Ada, sm_89, bf16) than the T4, and its 250 GB local NVMe can hold the 1.1 GB dataset. The fallback is g4dn.xlarge spot at about $0.25/h, cheaper per hour but with a 15–20% interruption rating. Which one costs less per *experiment* depends on how much faster the L4 is than the T4 on this CNN, which hasn't been measured.
- **Cost per experiment, at an assumed GPU wall-clock** (the M1→GPU speedup is not measured; a small CNN may be data-loader- or CPU-bound on a 4-vCPU box):

| GPU wall-clock incl. ~5 min setup | g4dn.xlarge spot (~$0.25) | g6.xlarge spot (~$0.40) | g6.xlarge on-demand Frankfurt ($1.006) |
|---|---|---|---|
| 20 min | $0.08 | $0.13 | $0.34 |
| 45 min | $0.19 | $0.30 | $0.75 |
| 2 h | $0.50 | $0.80 | $2.01 |

  At "a few experiments per day", that is roughly **$1–5/day on spot** and $3–15/day on-demand, plus a few cents of storage.
- g6f fractional-L4 sizes are the cheapest per hour, but I did not verify how much VRAM each slice gets. Treat them as an untested option, and check `nvidia-smi` and PyTorch support before relying on them.
- g4ad (AMD Radeon) appears in the price files at about $0.47/h in Frankfurt. It is not CUDA, so it is excluded.
- The 4 vCPUs on an xlarge may bottleneck PNG/TIF decoding and augmentation. If GPU utilization is low, g6.2xlarge (8 vCPU, $1.222 on-demand Frankfurt, <5% interruption) is the next step.

### Gaps
- No authoritative, dated current spot *price* for g6.xlarge or g4dn.xlarge in either region. The only reliable live source is `aws ec2 describe-spot-price-history --instance-types g6.xlarge g4dn.xlarge --product-descriptions "Linux/UNIX" --region eu-central-1`, which needs credentials (no AWS CLI is installed on this machine).
- Real M1-MPS → T4/L4 speedup for this model is unmeasured. It is the main driver of cost per experiment.

## Q2. New-account hurdles: GPU quotas, how to raise them, turnaround, approval odds

### Takeaway
On a new account, **every GPU quota defaults to 0 vCPUs**, and spot and on-demand are **separate quotas**. Expect to file two requests (8 vCPUs is enough for one g6.2xlarge or two xlarges), from the Service Quotas console or CLI, without a paid support plan. Turnaround runs from minutes to a few days. Small requests with a clear use case are approved most readily, but denials of new accounts do happen.

### Cited Findings
- Defaults from AWS docs: "Running On-Demand G and VT instances" = **0** (quota code `L-DB2E81BA`), "All G and VT Spot Instance Requests" = **0** (`L-3819A6DF`), On-Demand P = 0 (`L-417A185B`), Standard instances = 5. Quotas are in vCPUs, and "Amazon EC2 automatically increases your On-Demand Instance quotas based on your usage" — [EC2 instance type quotas](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html)
- A Spot quota does not cover On-Demand launches. A user with an approved Spot quota still got "You have requested more vCPU capacity than your current vCPU limit of 0 allows" when launching on-demand — [re:Post: Instance launch failed](https://repost.aws/questions/QUSPetOgBLTQuxHpy_ffhiAg/instance-launch-failed); [re:Post: g4dn quota error](https://repost.aws/questions/QUxBIRsixkTneUqAemd6QVjw/good-question-this-is-a-common-issue-when-you-try-to-launch-gpu-instances-like-g-4-dn-xlarge-in-aws-but-get-an-error-such-as-instance-limit-exceeded-you-do-not-have-sufficient-quota)
- Request the new *total*, not the increment. Smaller, staged requests are processed faster, "often within minutes to a few hours". Status shows under "Quota request history", and the result arrives by email — [RONIN blog, 2025-09-25](https://blog.ronin.cloud/how-to-request-an-ec2-quota-increase-on-aws-and-get-it-approved-faster/)
- A large jump, or a very new account, sends the request to human review. Approval for GPU spot quota can take days — [Meetrix guide](https://meetrix.io/blogs/increase-aws-vcpu-quota/); [GitHub issue: a 2026-10-19 request for 8 vCPU G/VT Spot in eu-central-1](https://github.com/zohars36/sizjobs/issues/7) (the outcome is not recorded)
- One forum report gives a 24–48 h turnaround — [NVIDIA forums](https://forums.developer.nvidia.com/t/aws-free-tier-account/55552). A request was denied after about 2 hours — [re:Post: EC2 vCPU limit issue](https://repost.aws/questions/QUlTBTbSCeRuC6-5HxxHvn4w/ec2-instance-creation-vcpu-limit-issue). There is also a fast.ai thread titled "AWS limit increase denied" — [fast.ai forums](https://forums.fast.ai/t/aws-limit-increase-denied/93874) (older)
- Quota requests don't require a paid support plan, and AWS may grant less than requested — [re:Post: how can I get GPU quota](https://repost.aws/questions/QU2TANoVHTTvOWbQ8-3te2oA/how-can-i-get-gpu-quota). Request states include PENDING, CASE_OPENED, APPROVED, DENIED, CASE_CLOSED — [Service Quotas API](https://docs.aws.amazon.com/servicequotas/2019-06-24/apireference/API_RequestedServiceQuotaChange.html)
- CLI: `aws service-quotas request-service-quota-increase --service-code ec2 --quota-code L-3819A6DF --desired-value 8 --region eu-central-1`, and the same with `L-DB2E81BA` for on-demand — [CLI reference](https://docs.aws.amazon.com/cli/v1/reference/service-quotas/request-service-quota-increase.html)
- Even with quota approved, a specific GPU type can be out of capacity in a given availability zone — [GMI Cloud blog](https://www.gmicloud.ai/en/blog/instant-gpu-instances-quota-reality) (vendor blog, weak evidence)

### Inferences
- File **both** requests (Spot G/VT and On-Demand G/VT, 8 vCPUs each) in **the region you'll use** on day one, because quotas are per region. Write a one-line use case ("single-GPU PyTorch training of a small CNN, one instance at a time, a few hours per day"). Having an on-demand quota lets the script fall back when spot capacity is short.
- Approval of a modest 4–8 vCPU request is likely but not guaranteed for a days-old personal account with little billing history. Plan for a 1–3 day wait. If denied, reply on the support case. It may help to run a few dollars of normal usage first so the account has some history.

### Gaps
- I found no systematic data on approval rates for *personal* versus business accounts, and no Reddit or HN reports from 2026 specifically.

## Q3. Billing granularity, storage and data-transfer costs for ~1.1 GB of data

### Takeaway
Linux EC2 is billed per second with a 60-second minimum, so the main cost is GPU hours. Storing 1.1 GB costs a few cents a month in S3, and pulling it into EC2 from S3 in the same region is free. Uploads are free, and the ONNX+JSON coming back is far under the 100 GB/month free egress. A persistent EBS volume costs single-digit dollars a month.

### Cited Findings
- "On-Demand Instances let you pay for compute capacity by the hour or second (minimum of 60 seconds)". Per-second billing covers Linux, Windows, RHEL and Ubuntu Pro, while SLES is billed per full hour — [EC2 On-Demand pricing](https://aws.amazon.com/ec2/pricing/on-demand/)
- "AWS customers receive 100 GB of free data transfer out to the internet free each month", aggregated across services and regions — [EC2 On-Demand pricing](https://aws.amazon.com/ec2/pricing/on-demand/)
- S3 Standard costs **$0.0245/GB-month in eu-central-1** and **$0.023/GB-month in us-east-1** (first 50 TB), per the AWS price list effective 2026-10-01 — [AWS S3 bulk price CSV, eu-central-1](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonS3/current/eu-central-1/index.csv); [us-east-1](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonS3/current/us-east-1/index.csv)
- gp3 EBS: AWS's worked example uses $0.08/GB-month and includes 3,000 IOPS and 125 MB/s baseline at no extra charge. The page does not name per-region prices — [EBS pricing](https://aws.amazon.com/ebs/pricing/)
- g6.xlarge and g5.xlarge include a 250 GB NVMe instance store, and g4dn.xlarge 125 GB — [Vantage g6](https://instances.vantage.sh/aws/ec2/g6.xlarge); [ec2.shop](https://ec2.shop) (specs only). The DLAMI mounts it at `/opt/dlami/nvme` — [DLAMI PyTorch 2.9 release notes](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-ami-gpupt29oss-ul2404-2026-10-05.html)

### Inferences
- Keeping 1.1 GB in S3 costs about **$0.03/month**. A 60 GB gp3 root volume kept for a stopped instance costs about **$4.8/month** at $0.08/GB, and a little more in Frankfurt (regional price not verified).
- The per-experiment code upload (a few hundred KB) and the ~3 MB result are both negligible. Uploading data into AWS is generally free.
- The cleanest data path: `aws s3 sync` the rasters to a bucket once, and have each instance `aws s3 sync s3://bucket/data /opt/dlami/nvme/data` at boot. That takes seconds to a minute for 1.1 GB within the region, and there is no persistent volume to pay for.

### Gaps
- Exact gp3 $/GB-month in eu-central-1, which I didn't pull from the 2026 price file. It is commonly cited around $0.095, but that is unverified here.

## Q4. Options compared on setup complexity from bash (raw EC2 spot, AWS Batch, SageMaker, stopped instance)

### Takeaway
For one job at a time launched from a laptop script, **raw EC2 via the AWS CLI** is the simplest and cheapest path. Launch a spot g6.xlarge from the DLAMI with `--instance-initiated-shutdown-behavior terminate`, pull data from S3, rsync the code, train, rsync the results back, then `shutdown -h`. Billing stops on termination. **SageMaker Training Jobs** cost about 40% more per hour, and its managed spot mode still runs at the SageMaker rate. AWS Batch adds setup work (compute environment, job queue, container image) that only pays off with many concurrent jobs.

### Cited Findings
- **SageMaker training list prices, effective 2026-10-01**, from the AWS price list CSV — [SageMaker eu-central-1](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonSageMaker/current/eu-central-1/index.csv); [us-east-1](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonSageMaker/current/us-east-1/index.csv):

| | ml.g4dn.xlarge | ml.g5.xlarge | ml.g6.xlarge | ml.g6e.xlarge |
|---|---|---|---|---|
| Frankfurt | $0.921 | $1.76 | $1.409 | $3.258 |
| us-east-1 | $0.736 | $1.408 | $1.127 | $2.61 |

  Compared with plain EC2 on-demand, that is +40% across the board (g6.xlarge Frankfurt: $1.409 vs $1.006). The price file lists "Managed Spot Training ml.g6.xlarge" at the same $1.409 list rate. The actual discount comes through `BillableTimeInSeconds`.
- Managed Spot Training: "can optimize the cost of training models up to 90% over on-demand". The savings formula is `(1 − BillableTimeInSeconds/TrainingTimeInSeconds)`. The SageMaker spot price "is a discount off the Amazon SageMaker AI on-demand price" and "is not comparable to the Amazon EC2 Spot price history". It "does not fall back to on-demand instances" and waits up to `MaxWaitTimeInSeconds`, which must be greater than `MaxRuntimeInSeconds`. Checkpoints are copied to and from S3, and checkpointing is recommended unless jobs are quick — [SageMaker Managed Spot Training docs](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html)
- SageMaker's free tier covers 50 h/month of m4/m5.xlarge training for the first 2 months. G-series instances are not included — [SageMaker pricing](https://aws.amazon.com/sagemaker/ai/pricing/)
- SageMaker also has a "Training Warm Pool" line item at the same hourly rate (e.g. ml.g6.xlarge $1.127 in us-east-1), billed while a warm pool keeps the instance alive between jobs — [SageMaker us-east-1 price CSV](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonSageMaker/current/us-east-1/index.csv)
- DLAMI ships the SSM agent, Docker and an NVMe mount point at `/opt/dlami/nvme` — [DLAMI release notes 20261003](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-ami-gpupt29oss-ul2404-2026-10-05.html)

### Inferences (comparison; setup effort is my judgement, not sourced)
| option | $/h (g6.xlarge Frankfurt) | setup from bash | per-job overhead | notes |
|---|---|---|---|---|
| **Raw EC2 spot + DLAMI** (`aws ec2 run-instances --instance-market-options MarketType=spot …`) | ~$0.40 spot / $1.006 OD | low: key pair, security group allowing SSH from your IP, IAM instance role for S3 read, one launch script | boot + `s3 sync` + venv activate, a few minutes (see Q5) | Cheapest. You handle interruption yourself (rerun the experiment, or checkpoint to S3). Use a one-time spot request with `InstanceInterruptionBehavior=terminate`. |
| **Stopped EC2 with persistent EBS** (`start-instances` / `stop-instances`) | same compute; EBS ~$5/mo while stopped | lowest per run: data and venv already on disk | boot only | On-demand only in practice. A persistent spot request can be stopped, but restarting depends on spot capacity. No per-run data copy. You pay for EBS around the clock, and a forgotten instance is the risk. |
| **SageMaker Training Job + Managed Spot** | $1.409 list, discounted by spot (savings vs SM OD, not EC2) | medium: execution role, S3 input channel, PyTorch estimator (Python SDK) or `aws sagemaker create-training-job` JSON with an AWS PyTorch DLC image URI | container pull and provisioning (minutes) | Auto-terminates, logs go to CloudWatch, artifacts land in S3 as model.tar.gz. Its billing model guarantees nothing is left running. |
| **AWS Batch on spot** | EC2 spot price (Batch itself is no extra charge) | high: compute environment, job queue, job definition, ECR image | ECS/instance scale-up, minutes | Built for queues of many jobs. Overkill for one at a time. |

- For this project's constraint (laptop is the only git writer; remote returns ONNX+JSON), raw EC2 maps directly onto an rsync-style flow: `rsync model/ pipeline/ → instance`, `ssh … train`, `rsync runs/<exp>/{*.onnx,*.json} ← instance`. SageMaker would mean packaging code via `source_dir` uploads and unpacking `model.tar.gz`.
- If you want SageMaker's "can't forget it running" property without its 40% markup, approximate it on EC2: launch with `--instance-initiated-shutdown-behavior terminate`, and put `shutdown -h +<maxmin>` first in user-data as a hard deadline.

### Gaps
- No sourced numbers for SageMaker container-start overhead or AWS Batch scale-up latency in 2026.
- The SageMaker managed-spot *effective* discount on g6 in Frankfurt is not published. It can only be seen per job.

## Q5. Cold-start / boot times, and Deep Learning AMI availability (PyTorch versions)

### Takeaway
AWS publishes current GPU PyTorch DLAMIs up to **PyTorch 2.14** (Ubuntu 26.04 and AL2023), released as recently as **2026-10-24**. All support G4dn, G5, G6, Gr6 and G6e. The latest AMI ID can be resolved from a public SSM parameter, which fits neatly into a bash launcher. CUDA now lives inside the PyTorch venv, so scripts must `source /opt/pytorch/bin/activate`. I found no sourced boot-time figures.

### Cited Findings
- Supported x86 GPU PyTorch DLAMIs: 2.14 (AL2023, Ubuntu 26.04), 2.13 (AL2023, Ubuntu 26.04), 2.12, 2.11, 2.10 (AL2023, Ubuntu 24.04). AWS warns: "PyTorch DLAMIs for versions 2.10 through 2.12 … contain known common vulnerabilities and exposures (CVEs) that cannot be patched … We recommend that you migrate to PyTorch 2.13 or later" — [DLAMI release notes index](https://docs.aws.amazon.com/dlami/latest/devguide/appendix-ami-release-notes.html)
- PyTorch 2.14 / Ubuntu 26.04 build 20261024: supported instances "G4dn, G5, G6, Gr6, G6e, P4, P4de, P5, P5e, P5en, P6-B200". NVIDIA driver 595.112.07, Python at `/opt/pytorch/bin/python`, gp3 root, NVMe at `/opt/dlami/nvme`. onnxruntime was *removed* from the AMI in this build — [release notes 20261024](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-ami-gpupt214oss-ul2604-2026-10-26.html)
- SSM lookup: `aws ssm get-parameter --region eu-central-1 --name /aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.14-ubuntu-26.04/latest/ami-id --query Parameter.Value --output text`. "CUDA is no longer available at the system level … activate the PyTorch virtual environment using `source /opt/pytorch/bin/activate`" — [DLAMI PyTorch 2.14 Ubuntu 26.04](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-gpu-pytorch-2.14-ubuntu-26-04.html)
- An older PyTorch 2.9 / Ubuntu 24.04 DLAMI is still being rebuilt (latest 20261003), with the same instance support list and onnx 1.23.1 included — [DLAMI PyTorch 2.9 20261003](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-ami-gpupt29oss-ul2404-2026-10-05.html)

### Inferences
- Pin the DLAMI's PyTorch version (or `pip install` exact versions of `onnx`, `onnxruntime` and the project's other deps into the venv at boot) so remote results stay comparable with local ones. The 2.14 AMI lacks onnxruntime, so the export/verification step must install it.
- A cold start from DLAMI plus a data pull plus pip installs typically takes several minutes. My rough working estimate is 2–5 min to SSH-ready on a large DLAMI root volume (first-boot EBS lazy loading from snapshot can slow initial reads), but this is **not sourced**. For the per-second bill this matters little (≈$0.03 at $0.40/h).
- If boot overhead becomes annoying: (a) a stopped instance with everything installed (boot only), or (b) bake a custom AMI with deps and data (`aws ec2 create-image`), keeping on-the-fly spot launches.

### Gaps
- No authoritative or recent measurements of DLAMI boot-to-SSH time, SageMaker training-job startup time, or EBS snapshot lazy-load impact on g6 in 2026.

## Q6. Free tier and credits: any free GPU? Credit programs for individuals?

### Takeaway
No GPU is free on AWS. Accounts created after 2025-07-15 get **up to $200 in credits** ($100 at sign-up and $100 more for completing activities) on a 6-month "Free plan". The Free plan blocks some services, and a GPU launch there may fail or require upgrading to the Paid plan, but remaining credits carry over. AWS Activate is for startups, so I found no credit route for an individual beyond this.

### Cited Findings
- New customers can get up to $200 in credits: "$100 upon sign-up and can earn an additional $100 in credits by using services such as Amazon EC2 and Amazon Bedrock"; the free plan lasts 6 months — [AWS What's New, 2025-07](https://aws.amazon.com/about-aws/whats-new/2025/07/aws-free-tier-credits-month-free-plan/); accounts before 2025-07-15 stay on the legacy model — [Neowin](https://www.neowin.net/news/aws-free-tier-overhauled-new-users-get-200-in-credits-but-theres-a-catch/)
- The Free plan is "limited from accessing a subset of AWS services and offerings that would immediately consume the entire Free Tier credit amount or require hardware purchases". On upgrade to Paid, "your remaining Free Tier credits will automatically apply to future AWS bills until they expire". At expiry the account is suspended, and "If you don't upgrade your account within 90 days, AWS will permanently erase your AWS account and all its content". The FAQ gives conflicting expiry periods for credits (12 months in one answer, 6 in another) — [AWS Free Tier FAQs](https://aws.amazon.com/free/free-tier-faqs/)
- SageMaker's free tier covers only m4/m5.xlarge — [SageMaker pricing](https://aws.amazon.com/sagemaker/ai/pricing/). The legacy 12-month EC2 free tier covered only t2/t3.micro — [GMI Cloud blog](https://www.gmicloud.ai/blog/best-cloud-gpu-providers-with-free-credits-trials) (vendor blog)

### Inferences
- With ~$0.40/h spot and a few runs a day, $200 of credits would cover roughly **200–500 GPU-hours**, i.e. weeks to months of this project, *if* the credits apply to G-instance usage. That is not confirmed (see Gaps).
- Expect to need the **Paid plan** before GPU quota requests or launches work. Upgrading keeps the credits.

### Gaps
- No AWS page explicitly says whether Free-plan credits can be spent on G-family EC2, or whether Free-plan accounts can request G quotas. The FAQ only references an "eligible services" list I did not retrieve.
- AWS Activate eligibility for individuals was not researched further. It targets startups, often via accelerator or VC affiliation.

## Q7. Safety: avoiding a forgotten running instance

### Takeaway
Use several layers. In the instance, set `--instance-initiated-shutdown-behavior terminate` plus a `shutdown -h +N` deadline in user-data, and end the train script with `sudo shutdown -h now`. In the launcher, use a bash `trap` that calls `terminate-instances`. At the account level, set an AWS Budget with email alerts (free) and optionally a budget action. Both the spot one-time request and SageMaker terminate on their own.

### Cited Findings
- AWS Budgets: you can "monitor and receive notifications on your budgets free of charge". The first two action-enabled budgets are free each month, then $0.10/day each. Emailed reports cost $0.01 each — [AWS Budgets pricing](https://aws.amazon.com/aws-cost-management/aws-budgets/pricing/)
- Example of a personal-account setup (Oct 2026): a $50 monthly budget with alerts at 30/60/100% plus forecast, and daily Cost Anomaly Detection email — [GitHub issue zohars36/sizjobs#7](https://github.com/zohars36/sizjobs/issues/7)
- SageMaker managed spot jobs stop at `MaxWaitTimeInSeconds` and never fall back to on-demand — [SageMaker Managed Spot Training docs](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html)

### Inferences (standard AWS mechanisms; behaviours below are from general AWS knowledge, not fetched this session)
1. Launch flags: `--instance-initiated-shutdown-behavior terminate` (an OS shutdown then *terminates* the instance and deletes the root EBS if DeleteOnTermination=true, the DLAMI default). Add `--tag-specifications` with a project tag for cleanup scripts.
2. A hard deadline in user-data: `shutdown -h +300` (5 h), so even a hung job or a lost laptop connection ends billing.
3. Laptop side: `trap 'aws ec2 terminate-instances --instance-ids $IID' EXIT INT TERM` in the launcher. Run a sweep at each loop iteration: `aws ec2 describe-instances --filters Name=tag:project,Values=ballpark Name=instance-state-name,Values=running` and terminate anything older than the expected maximum.
4. Account side: a monthly AWS Budget (e.g. $30) with email at 50/80/100% actual and 100% forecast, and Cost Anomaly Detection. Optionally a CloudWatch alarm on `CPUUtilization < 5% for 30 min` with the EC2 *stop/terminate* alarm action, as an idle killer.
5. Use a dedicated IAM user or role for the loop with permissions limited to ec2 run/terminate/describe on tagged resources and S3 read for the data bucket. No root keys.

### Gaps
- I did not fetch the AWS docs page for CloudWatch alarm EC2 actions or for instance-initiated shutdown behaviour this session. The mechanisms are long-standing, but quote them from the docs before relying on exact semantics.
