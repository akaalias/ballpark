# Free-tier and serverless GPU platforms for unattended, scripted PyTorch training (researched 2026-10-31)

Context: a bash loop on a MacBook launches one training job per experiment. Each job uses a small CNN, takes 45 min to 4 h on an M1, reads about 1.1 GB of rasters that rarely change, and runs code that changes every time. The remote has to run without anyone at the keyboard and return an ONNX file plus a JSON. All prices below are list prices read on 2026-10-31 unless another date is given.

## 1. Modal: what it costs now, and is paying per second a reasonable answer?

### Takeaway
Modal Starter is still $0 per month and includes $30 of compute credit each month. GPU time is billed per second. A single L4 costs about $0.80/h and an A100-40GB about $2.10/h, plus CPU and memory. With 3–10 jobs a day, the free credit lasts days, not a month. Paying is cheap per job (well under $1–3 for a 30–60 min A100 run), and Modal fits the job shape better than anything else here: Python SDK, volumes, timeouts up to 24 h, nothing to babysit. It is the reasonable paid baseline. The real cost question is how fast the job runs on a GPU, which nobody has measured yet.

### Cited Findings
- Starter plan: $0 plus compute, with **$30/month in free credits**, 3 seats, 100 containers and a GPU concurrency of 10. Team plan: $250/month plus compute, with $100/month in credits. Enterprise is custom. The page calls the credits a monthly allotment but does not say whether unused credit rolls over. — [Modal pricing](https://modal.com/pricing) (read 2026-10-31)
- GPU list prices per second: T4 $0.000164, L4 $0.000222, A10 $0.000306, L40S $0.000542, A100-40GB $0.000583, A100-80GB $0.000694, H100 SXM $0.001097, H200 $0.001261, B200 $0.001736. — [Modal pricing](https://modal.com/pricing)
- CPU costs $0.0000131 per physical core per second (1 core = 2 vCPU). Memory costs $0.00000222 per GiB per second. Volumes cost $0.09/GiB/month with 1 TiB/month free. Egress costs $0.04/GiB. Sandboxes are billed at 3× list price. — [Modal pricing](https://modal.com/pricing)
- A change to the Starter plan (token billing from the first request) "took effect September 1, 2026". This looks like it concerns Modal's inference/token products rather than GPU functions. — [Modal pricing](https://modal.com/pricing)
- Credit programmes: startups can get free credits (no amount stated). **Academics (graduate students, labs, researchers) can get up to $10k.** AWS, GCP or Azure credits cannot be applied, although eligible AWS Activate startups may get extra Modal credits. — [Modal pricing](https://modal.com/pricing); the $10k academic figure is also reported by [gpuperhour, 2026-09-21](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)
- The default function timeout is 300 s. It can be set anywhere from 1 s to **24 h**. Each retry gets a fresh timeout. — [Modal docs: timeouts](https://modal.com/docs/guide/timeouts)

### Inferences
- Converted to hourly rates: T4 ≈ $0.59/h, L4 ≈ $0.80/h, A10 ≈ $1.10/h, L40S ≈ $1.95/h, A100-40 ≈ $2.10/h, A100-80 ≈ $2.50/h, H100 ≈ $3.95/h. Add CPU and memory. For example, 8 cores and 32 GiB is about $0.38/h + $0.26/h ≈ $0.64/h, which is not negligible next to an L4.
- $30 of credit buys roughly 20–35 L4-hours or about 12 A100-hours per month, depending on how much CPU and RAM the job requests. A loop that runs for days burns through that within the first days of the month. After that it is pay-per-use.
- Rough per-experiment cost if the GPU speeds the job up 3–10× over the M1 (an assumption, not measured): a 45 min M1 job becomes 5–15 min on an A10/L4, which is about $0.10–0.40. A 4 h M1 job becomes 25–80 min, which is about $0.50–1.50 on an L4 or $1–3 on an A100. At 5–10 experiments a day that comes to about **$50–300/month**, so paying is reasonable. A small CNN probably will not use an A100 fully. Benchmark L4, A10 and A100 once and choose by cost per experiment, not by price per hour.
- The 1.1 GB dataset fits in a Modal Volume at no extra cost (well under the 1 TiB free). The loop already used Modal, so the harness can be revived as it is.
- The **$10k academic grant** is the most valuable lever here. The user is an individual open-source researcher, not a graduate student, so eligibility is uncertain. It is worth applying.

### Gaps
- Whether unused monthly credit rolls over (the pricing page does not say).
- What evidence the academic grant requires, and whether an unaffiliated individual qualifies.
- Cold-start time for a GPU container with a cached image. Not found in this pass; it is typically seconds to under a minute, but that is unverified.

## 2. Modal-like serverless platforms

### Takeaway
Hugging Face Jobs is the closest drop-in alternative. It is pay-as-you-go with a `docker run`-style CLI, local directories synced to a bucket, a blocking `hf jobs wait`, and T4 at $0.40/h, L4 at $0.80/h and A10G at $1.00/h, which matches or undercuts Modal. Cerebrium lists the same per-second prices as Modal, with a slightly cheaper A100 and H100. Beam is cheap on RTX 4090 ($0.69/h serverless) but costs $20/month. RunPod Serverless is built for request/response endpoints, not batch training, while RunPod pods (Community 4090 at $0.34/h) are the cheapest raw GPU time. Koyeb, fal, Replicate and Baseten are built for inference and fit this job poorly. None of these platforms gives a meaningful *recurring free* GPU allowance except Modal ($30) and Beam ($20, but Beam charges a $20 plan fee).

### Cited Findings
**Hugging Face Jobs**
- Available to any user or org with a positive credit balance, billed per second as pay-as-you-go. Run with `hf jobs run --flavor a10g-small <image> <cmd>` or `hf jobs uv run script.py`. — [HF hub docs: Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)
- Hardware prices per hour: t4-small $0.40, t4-medium $0.60, l4x1 $0.80, a10g-small $1.00, a10g-large $1.50, l40sx1 $1.80, a100-large (80 GB) $2.50, rtx-pro-6000 $2.75, h200 $5.00. — [HF hub docs: Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)
- **The default timeout is 30 min.** Training needs `--timeout` or `timeout="4h"`, and units up to days (`1d`) are accepted. — [HF hub docs: Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)
- Data: `sync_job_volume("./training-data", "/data")` uploads to a `jobs-artifacts` Storage Bucket. Re-syncs upload only new or modified files. The CLI form is `-v ./local:/data`. Outputs go to a read-write volume and come back with `sync_bucket`. — [HF hub docs: Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)
- `hf jobs wait <id>` blocks and exits 0 only on success. A non-detached `hf jobs run` also returns non-zero on failure, so `&&` chaining works in bash. Retries (`attempts=`), secrets and env vars are supported. — [HF hub docs: Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)

**Cerebrium**
- Hobby plan is free plus compute (3 apps, 5 concurrent GPUs). Standard is $100/month. Per-second prices: T4 $0.000164, L4 $0.000222, A10 $0.000306, L40S $0.000542, A100-40 $0.000555, A100-80 $0.000583, H100 $0.000944. Storage is $0.05/GB/month with the first 100 GB free. The page lists no free credit amount and no maximum run duration. — [Cerebrium pricing](https://www.cerebrium.ai/pricing) (read 2026-10-31)
- Third-party sources disagree on the free credit: one says up to $1,000 of onboarding credit, another says there is no free tier. — [CostBench](https://costbench.com/software/ai-model-hosting/cerebrium/free-plan/); treat as unverified.

**Beam (beam.cloud)**
- The Developer plan is **$20/month and includes $20 of credit each month**. Unused monthly credit expires at renewal. Team plan is $89/month. — [Beam pricing](https://www.beam.cloud/pricing) (read 2026-10-31)
- Serverless prices: RTX 4090 $0.000192/s (≈$0.69/h), RTX 5090 ≈$1.09/h. On-demand machines: H100 PCIe from $1.83/h, A100-80 SXM from $1.36/h, 4090 from $0.44/h. Storage up to 1 TB is included. T4 and A10G are not listed, and neither is a maximum task duration. — [Beam pricing](https://www.beam.cloud/pricing)

**RunPod**
- Serverless prices (page updated 2026-10-08): 16 GB class $0.58/h, 24 GB L4/A5000/3090 $0.75/h, 4090 $1.34/h, 48 GB A6000/A40 $1.22/h, L40S $1.75/h, A100-80 $2.72/h, H100 $5.39/h. — [RunPod pricing](https://www.runpod.io/pricing)
- Pod prices: RTX 4090 $0.34/h Community or $0.89/h Secure, A100 PCIe $1.19/$1.79, H100 SXM $2.69/$3.99. Network storage is $0.07/GB/month. The page mentions no free credits. — [RunPod pricing](https://www.runpod.io/pricing)

**Koyeb**
- Serverless GPU prices are per second with scale-to-zero. After the November 2025 cut: A100 $1.60/h and H100 $2.50/h. December 2025 figures: L4 $0.70/h, L40S $1.55/h. A January 2026 Koyeb post still lists A100 at $2/h, so the sources conflict. — [Koyeb blog, price cut](https://www.koyeb.com/blog/koyeb-serverless-gpus-slashing-prices-on-a100-h100-and-l40s-up-to-24); [Koyeb blog, Jan 2026](https://www.koyeb.com/blog/best-serverless-gpu-platforms-for-ai-apps-and-inference-in-2026)
- A startup programme of up to $30k in credits exists. — [gpuperhour, 2026-09-21](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)

**fal / Replicate / Baseten**
- fal: a 2026 comparison quotes an A100 from $0.99/h and an H100 from $1.99/h for fal compute but says sales contact is required. The figures on that page are ambiguous. — [Koyeb blog comparison, Jan 2026](https://www.koyeb.com/blog/best-serverless-gpu-platforms-for-ai-apps-and-inference-in-2026)
- Replicate and Baseten: no current training-pricing data found in this pass.

**Lightning AI**
- Free tier: **up to 30 credits to start, which is not a monthly allowance.** That is 5 credits on registration plus 25 more on adding a card, which the FAQ says is about 80 GPU-hours on interruptible machines. **Unused credits expire after 12 months.** Phone verification is required. — [Lightning pricing](https://lightning.ai/pricing) (read 2026-10-31)
- Listed prices: T4 $0.55/h, L4 $0.79/h, L40S $3.54/h, A100-40 $2.19/h, A100-80 $1.85/h, H100 $5.45/h. The page shows interruptible T4 *higher* than on-demand ($0.61–0.73), which looks like a page error. The pricing page does not mention a jobs API for the free tier. — [Lightning pricing](https://lightning.ai/pricing)
- The free plan allows "1 free active Studio (4 hour restarts)" with 50 GB of persistent storage. — [gpuperhour, 2026-09-21](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)

### Inferences
- **HF Jobs is the best-documented fit for "bash triggers, blocks, pulls back outputs"**: `hf jobs uv run -v ./data:/data -v ./out:/out:rw --flavor l4x1 --timeout 5h train.py && …`. The 1.1 GB dataset is synced once and reused, because only changed files upload. It is cheaper than Modal for T4 ($0.40 vs ~$0.59 + CPU) and on par for L4. CPU and RAM come bundled in the flavor price, which makes the job cheaper than Modal once Modal's separate CPU and memory charges are added.
- Cerebrium's prices match Modal's exactly (T4, L4, A10, L40S) but its free credits are unconfirmed. Its tooling is built around deployed apps (endpoints), so a batch training job means writing a run-once endpoint.
- RunPod Serverless exposes request-handler endpoints, and its serverless prices are higher than its pod prices. For training, a pod started and stopped via `runpodctl` or the API is cheaper (a Community 4090 at $0.34/h), but the user already abandoned RunPod, so the reason it was abandoned matters more than the price.
- Lightning's free credit is a one-off ~$30 starter balance, effectively the same as one month of Modal Starter.

### Gaps
- Cold-start times for each platform (no primary measurements found).
- Maximum job duration on Beam and Cerebrium (not stated on their pricing pages).
- Whether Lightning's free tier can launch jobs from the CLI (`lightning run job`) or only through Studios.

## 3. Free notebook platforms: can they be automated?

### Takeaway
**Kaggle is the only free option with an official, documented non-interactive path.** The CLI pushes a kernel with a chosen accelerator (T4×2, L4, P100-class and others), runs it in the background, reports its status and downloads its outputs. The rarely changing 1.1 GB of rasters goes up once as a private Kaggle Dataset. The catch is a weekly quota of about 30 GPU-hours and 12 h sessions, which covers roughly 10–40 experiments a week depending on how fast the GPU is. Colab cannot be automated legitimately on the free tier, because its FAQ explicitly bans bypassing the notebook UI and remote control without paid compute units. SageMaker Studio Lab is closed to new users. Paperspace's free tier is an 8 GB M4000, slower than the M1 in any practical sense.

### Cited Findings
**Kaggle**
- Official CLI: `kaggle kernels push -p <folder>` uploads and runs the kernel. `--accelerator NvidiaL4` (or `NvidiaTeslaT4` [T4×2, the default GPU], `NvidiaTeslaA100`, `NvidiaL4X1`, `NvidiaH100`, `NvidiaRtxPro6000`, `TpuV5E8`, `TpuV6E8`) selects the hardware, and `-t/--timeout <s>` caps the run. **"Some accelerators are limited to participants of specific competitions, and some are limited to Kaggle admins."** — [Kaggle API docs: kernels](https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels.md)
- `kaggle kernels status owner/slug` reports running, completed or failed. `kaggle kernels output owner/slug -p <dir> --file-pattern <regex>` downloads the outputs. Kaggle Secrets can only be set on kaggle.com, not through the CLI. — [Kaggle API docs: kernels](https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels.md)
- One community script found that when the accelerator string doesn't match exactly, the CLI silently falls back to the default accelerator. Validate this in the harness. Third-party wrappers (kagglex, the Dify plugin, the kaggle-kernel-actions GitHub Action) all automate the same create-once, push, poll, download pattern. kagglex notes that the public API cannot cancel a running kernel. — [kagglex on PyPI](https://pypi.org/project/kagglex/); [kaggle-dify-plugin](https://github.com/ki3nd/kaggle-dify-plugin); [HF Space push script](https://huggingface.co/spaces/cuilabs/bee/blob/main/scripts/push_kaggle_tpu_kernel.py)
- Limits: "12 hours execution time for CPU and GPU notebook sessions", 9 h for TPU. The weekly GPU quota appears in account settings. It is commonly about 30 h, and Kaggle's docs give no number. Linking Colab Pro or Pro+ adds 15 or 30 GPU-hours a week. — [gpuperhour, 2026-09-21, citing Kaggle docs](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)
- Users report a "floating" quota of at least 30 h that resets on Saturday at midnight UTC, with 30–41 h seen. Other sources give a 9 h GPU session limit, but 12 h has the stronger support. — [search synthesis, aimultiple / gpuperhour](https://aimultiple.com/free-cloud-gpu)
- Kaggle's product lead: GPUs are free but "not unlimited", and Kaggle "continually strive[s] to eliminate abuse". — [NVIDIA blog](https://developer.nvidia.com/blog/how-kaggle-makes-gpus-accessible-to-5-million-data-scientists/)

**Google Colab**
- On free runtimes without a positive compute-unit balance, the following are disallowed: "remote control (such as SSH shells or remote desktops), bypassing the notebook UI in favor of a web UI… running distributed computing workers". Sessions can be terminated without warning. Free sessions last at most 12 h. Limits are dynamic and unpublished. Pro+ allows up to 24 h of continuous execution. — [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- Pay As You Go costs $9.99 for 100 compute units (about $0.10 per unit). Units reportedly expire after 90 days. Measured burn rates (March 2026): T4 1.19 units/h (≈$0.12/h), L4 1.71 (≈$0.17/h), A100-40 5.40 (≈$0.54/h), A100-80 7.52 (≈$0.75/h). Older sources give much higher rates (T4 ≈1.96, A100 ≈15), so the figures conflict. — [McCormick, checked 2026-03-10](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/); [Thundercompute](https://www.thundercompute.com/blog/google-colab-alternatives)
- Colab Enterprise (a GCP product billed per hour): T4 ≈$0.42/h, L4 ≈$0.67/h, A100 ≈$3.52/h, plus CPU and RAM. — [Colab Enterprise pricing](https://cloud.google.com/colab/pricing)

**SageMaker Studio Lab**
- "No longer open to new customers." Existing users keep 4 h GPU sessions with a limit of 4 h per 24 h. — [gpuperhour, 2026-09-21](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)

**Paperspace Gradient**
- Free plan: an M4000 (8 GB, 2015-era), 5 GB of storage, auto-shutdown after at most 6 h. The page was last verified 2026-07-13, and whether capacity is actually available was not tested. — [gpuperhour, 2026-09-21](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)

### Inferences
- **The Kaggle workflow for this loop:** (1) `kaggle datasets create` once with the rasters as a *private* dataset, then `kaggle datasets version` only when they change. (2) For each experiment, rewrite a kernel folder holding `train.py` (the current `model/` code is bundled or inlined) and `kernel-metadata.json`, which references the dataset in `dataset_sources`, is private, and has GPU enabled. (3) Run `kaggle kernels push -p dir --accelerator NvidiaTeslaT4`. (4) Poll `kaggle kernels status` until complete or error. (5) `kaggle kernels output … --file-pattern '\.(onnx|json)$'`. Every step is documented CLI, with no browser. The metadata keys (`dataset_sources`, `enable_gpu`, `is_private`, `enable_internet`) are documented in `kernels_metadata.md` in the same repo. I did not fetch that file, so verify the exact keys.
- The T4×2 default is roughly in M1-GPU territory or a few times faster for a small CNN. That is an educated guess and needs a benchmark. Kaggle's main win is that it is *free and parallel to the laptop*, not that it is faster. The L4 option (if it is not competition-restricted) is meaningfully faster than a T4.
- At about 30 GPU-h a week, a 12 h session cap and 1–4 h per experiment, Kaggle supports about 7–30 experiments a week. That is about one day of loop throughput, so it works best as a supplement, or as the main engine with the loop sleeping until Saturday's quota reset.
- ToS risk: no explicit Kaggle rule against API-driven runs was found. The API exists specifically for this. The risk is "abuse" heuristics and silent quota floating. Keep kernels private, use one account, and do not rotate accounts to multiply quota.
- Colab: automating a **free** Colab session violates the FAQ (UI bypass). With paid units, the restriction lifts, but there is still no official headless job API for consumer Colab. Colab Enterprise has one but is plain GCP billing at prices no better than Modal or HF.

### Gaps
- Kaggle's official notebook docs page (kaggle.com/docs/notebooks) renders through JavaScript and could not be fetched. The quota, phone-verification requirement (needed for GPU and internet, per long-standing practice), `/kaggle/working` output size (~20 GB) and private dataset size limits are **unverified** in this pass.
- Whether L4 and H100 accelerators are generally available to ordinary users or restricted to competitions.
- No primary Kaggle ToS clause on automated use was found.

## 4. Research and credit programmes open to an individual open-source researcher in Germany

### Takeaway
The realistic programmes are **Modal's academic grant (up to $10k)** and **Google TPU Research Cloud** (free TPUs, but that means porting PyTorch to XLA). Startup programmes (NVIDIA Inception, AWS Activate, Google for Startups, Microsoft for Startups, Koyeb) need an incorporated company. Hyperscaler free trials mostly exclude GPUs.

### Cited Findings
- **TPU Research Cloud:** free access for accepted researchers to a cluster of more than 1,000 Cloud TPU devices. Supports TensorFlow, PyTorch, JAX and Julia. Accepted researchers are expected to share their work publicly (papers, **open-source code**, blog posts), give feedback, and follow the AI principles. Duration and whether institutional affiliation is required are not stated. Entry is through an "express interest" form. — [TRC about](https://sites.research.google/trc/about/)
- **Modal academic grant:** up to $10k for graduate students, labs and researchers. — [Modal pricing](https://modal.com/pricing)
- **NVIDIA Inception:** requires an incorporated company under 10 years old with a working website. It offers partner credits with no stated amount. **AWS Activate:** $1k for founders, needs a paid account and pre-Series-B status. **Google for Startups:** $2k for an MVP, up to $350k for AI-first seed to Series A. **Microsoft for Startups:** up to $150k in Azure. **Koyeb:** up to $30k. — [gpuperhour, 2026-09-21](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)
- Hyperscaler trials: GCP gives $300 for 90 days but **no GPUs on trial VMs**. AWS gives up to $200 for select services. Azure gives $200 for 30 days, with GPU quota unclear. — [gpuperhour, 2026-09-21](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)

### Inferences
- TRC fits poorly. PyTorch/XLA on TPU adds a porting and debugging burden to every experiment the agent writes, and TPUs suit large batch matmul work better than a small CNN with custom ops.
- Modal's grant is the best application to make. The project is open source (`akaalias/ballpark`), has a public research log and a lineage database, which makes a good case, and it reuses the existing Modal harness.
- Startup programmes need a company (a GmbH or UG, or a sole proprietorship with a website for Inception). That is not worth setting up for the credits alone.

### Gaps
- No current details found on a Lambda research grant or an NVIDIA academic grant programme. Neither came up in this pass.
- No Germany- or EU-specific programmes (for example EuroHPC access calls or GWDG/NHR) were investigated. NHR/EuroHPC calls usually need an academic affiliation, which is unverified here.

## 5. Summary table (late October 2026)

| Platform | Free recurring GPU? | Cheapest useful GPU | Scripted/unattended? | Dataset persistence | Max job | ToS for automation |
|---|---|---|---|---|---|---|
| Modal | $30/mo credit | T4 ≈$0.59/h, L4 ≈$0.80/h (+CPU/RAM) | Yes (Python SDK) | Volume, 1 TiB free | 24 h | Fine |
| HF Jobs | No (pay as you go) | T4 $0.40/h, L4 $0.80/h, A10G $1.00/h (all-in) | Yes (`hf jobs run/wait`) | Storage bucket via `sync_job_volume` | timeout settable in days, 30 min default | Fine |
| Cerebrium | Unclear (credits unconfirmed) | Same as Modal; A100-80 $2.10/h | Endpoint-shaped | 100 GB free | not stated | Fine |
| Beam | $20/mo credit but $20/mo fee | 4090 ≈$0.69/h | Yes | 1 TB included | not stated | Fine |
| RunPod | No | Pod 4090 $0.34/h (Community) | API/pods; serverless is endpoint-shaped | Network volume $0.07/GB/mo | n/a | Fine |
| Lightning | ~$30 one-off, expires after 12 mo | T4 $0.55/h | Studios; jobs API unclear on free tier | 50 GB | 4 h restarts (free) | Fine |
| **Kaggle** | **~30 GPU-h/week free** | T4×2 (L4 listed) | **Yes, official CLI push/status/output** | Private Kaggle Dataset | 12 h session | No explicit ban found; "abuse" monitored |
| Colab free | Dynamic, unpublished | T4 | **No: FAQ bans UI bypass on free tier** | Drive (manual) | 12 h | Violates FAQ |
| Colab PAYG | No ($9.99/100 units) | T4 ≈$0.12/h, A100 ≈$0.54/h (measured March 2026) | No official headless API | Drive | 24 h (Pro+) | Allowed with units, but no job API |
| Studio Lab | Closed to new users | — | — | — | 4 h/day | — |
| Paperspace free | M4000 8 GB | — | Weak | 5 GB | 6 h | — |
