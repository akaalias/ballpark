# GPU rental marketplaces and neoclouds for short, scriptable PyTorch training jobs (as of 2026-10-31)

Workload: one small CNN training run at a time, a few per day. Each run takes 45 min to 4+ h on an M1. The dataset is about 1.1 GB, rarely changes, and should stay on the remote. The code changes every run. The laptop is the only git writer. The remote trains and returns ONNX + JSON. The user is in Germany. RunPod (an always-on pod) and Modal (until credits ran out) were already tried.

Prices are on-demand, per GPU, per hour, in USD, and exclude VAT. Each price carries the date it was fetched. "2026-10-31" means fetched today from the provider's page or a tracker that showed today's date.

## Comparison table (cheapest suitable GPUs, billing, scriptability, persistent storage)

### Takeaway
For a small CNN, the cheapest workable options are RTX 3090/4090-class cards on marketplaces: Vast.ai about $0.20–0.35, RunPod Community $0.22/$0.34, Salad $0.16–0.33, TensorDock about $0.35–0.46. All are billed per second or per minute. The cleanest way to get "create, run, destroy, pay only while training" with a persistent dataset is RunPod: a network volume ($0.07/GB/mo, so about $0.08/mo for 1.1 GB) plus a fresh pod per run, or RunPod Serverless. Vast.ai is cheapest but has host variance and machine-bound storage. Lambda, DigitalOcean and CoreWeave cost more and don't fit this workload as well.

### Cited Findings

| Provider | Cheapest suitable GPU(s), $/GPU-h (date) | Billing granularity / minimum | Official CLI/API | Persistent storage & idle cost | Notes |
|---|---|---|---|---|---|
| **RunPod Pods** | Community Cloud: 3090 $0.22, 4090 $0.34, 5090 $0.69, A40 $0.35, L4 $0.44, A100 PCIe $1.19, H100 PCIe $1.99. Secure Cloud: 3090 $0.50, 4090 $0.89, 5090 $1.19, A100 $1.79, H100 SXM $3.99 (2026-10-31) | Per-hour or per-second toggle | Yes (runpodctl, API, SkyPilot) | Network volume $0.07/GB/mo (<1 TB). Volume disk $0.10/GB/mo running, **$0.20/GB/mo while stopped** | Community 4090 at $0.34 showed **out of stock** today; the cheapest in-stock 1x 4090 was Secure at $0.89 (2026-10-31) |
| **RunPod Serverless** | 24 GB class (3090/L4/A5000) $0.75/h, 4090 $1.34/h, 5090 $1.79/h, A100 $2.72/h, H100 $5.39/h (2026-10-31) | Per second, rounded up; idle timeout (default 5 s) billed; scaled-down workers free | Yes (API) | Network volume same $0.07/GB/mo | Roughly 2–4x the Community pod price. Default spend limit $80/h |
| **Vast.ai** | 4090 about $0.20–0.35 (cheapest offer about $0.20, median $0.48; tracker "from $0.309", Aug 2026). No official Oct 2026 numbers found | Per second. Interruptible tier "50%+ cheaper" | Yes (CLI, Python SDK, API, SkyPilot) | Volumes exist but are **tied to one physical host**. Storage is billed while stopped | Prepaid credits; crypto accepted. Restart can fail if the GPU was re-rented |
| **Thunder Compute** | A6000 48 GB $0.35, L40 $0.79, A100 80 GB $1.09, H100 $3.20 (instances). Sandboxes: A100 $1.95, H100 $2.95 (2026-10-31) | Instances per minute, only while running. Sandboxes per second | Yes (`tnr` CLI, REST API, MCP, VS Code extension) | Disk $0.219/GB/mo, first 100 GB/GPU included and billed only while running. Snapshots $0.05/GB/mo. Egress $0 | No consumer RTX cards listed |
| **Lambda Cloud** | A6000 $1.09, A10 $1.29, A100 40 GB $1.99, GH200 $2.29, H100 PCIe $3.29, H100 SXM $4.29 (2026-10-31) | Not stated on the pricing page | API/CLI exist (SkyPilot supports it) | Filesystem price not on the pricing page | Self-serve, first-come. Availability is often the problem (see Gaps) |
| **TensorDock** | 4090 $0.35–0.37 on its own pages. Tracker: from $0.46 (2026-10-31), $0.25 (gpuperhour), marketplace lows $0.15–0.18 (Mar 2026) | Hourly marketplace | API (not in SkyPilot) | n/a | A competitor reports empty/unresponsive availability since late 2025. Another source says it was acquired by Voltage Park |
| **Salad** | 4090: high priority $0.33, medium $0.27, low $0.22, lowest $0.16 (2026-10-31) | n/a; not billed while a node is down or being reallocated | Portal (API exists; not confirmed here) | **Ephemeral**; no persistent volumes described | Container-group model on consumer PCs. No SSH documented. Nodes get reallocated, so checkpointing is needed |
| **Hyperbolic** | Caps: H100 SXM $3.20, H100 PCIe $3.00, 4090 $0.50. July/Aug 2026 rate hike; the 4090 may have been **dropped** | n/a | n/a | n/a | Platform takes 10% of rental income. Now H100/H200/B200 focused |
| **DigitalOcean GPU Droplets** (Paperspace's parent) | RTX 4000 Ada $0.76, RTX 6000 Ada $1.57, L40S $1.57, H100 $4.41 (prices effective 2026-08-01) | Per second, **5-min minimum** | Yes (doctl/API; Paperspace in SkyPilot) | **Powered-off droplets are still billed in full**; you must destroy them | No A100 listed |
| **Jarvislabs** | L4 $0.44 (spot $0.29), A30 $0.41, A100 40 GB $0.89, A100 80 GB $1.49 (spot $0.89), H100 $3.49 (spot $1.54) (2026-10-31) | Per minute, pause/resume | CLI listed | Disk/filesystem $0.10/GB/mo on provisioned capacity | India regions mentioned. Billed in USD outside India |
| **Prime Intellect** | Older listings: 4090 $0.35, A100 $0.87, H100 from $1.65. Sep 2024: H100 $1.49, A100 $0.79. Aggregator: H100 about $2.43 | n/a | Marketplace aggregator; in SkyPilot | n/a | **No current official price sheet found** |
| **Shadeform** | Aggregates 30+ clouds; no prices shown on the homepage | n/a | CLI + MCP server; in SkyPilot | Depends on the underlying cloud | Centralized billing. Reserved has a 12-month minimum |
| **CoreWeave** | Mostly 8-GPU H100/A100 blocks; GH200 is the single-GPU exception | Reported 1-week minimums (third-party) | Kubernetes/Slurm | — | **Not suitable** (see below) |

Sources for the table:
- RunPod pod, serverless and storage prices — [RunPod pricing](https://www.runpod.io/pricing)
- RunPod serverless billing semantics — [RunPod serverless pricing docs](https://docs.runpod.io/serverless/pricing)
- RunPod 4090 stock status today — [GetDeploying RTX 4090](https://getdeploying.com/gpus/nvidia-rtx-4090)
- Thunder Compute — [Thunder Compute pricing](https://www.thundercompute.com/pricing)
- Lambda — [Lambda pricing](https://lambda.ai/pricing)
- Vast.ai billing and CLI — [Vast.ai pricing](https://vast.ai/pricing)
- Vast.ai storage — [Vast.ai storage docs](https://docs.vast.ai/documentation/instances/storage/types)
- Vast.ai 4090 prices — [computeprices Vast 4090](https://computeprices.com/providers/vast/gpus/rtx4090); [Vast.ai review 2026](https://www.fast.io/resources/vast-ai-review-2026.md)
- TensorDock — [TensorDock cloud GPUs](https://tensordock.com/cloud-gpus); [TensorDock 4090](https://tensordock.com/gpu-4090); [Spheron TensorDock pricing 2026](https://www.spheron.network/blog/tensordock-pricing-2026/); [gpuperhour TensorDock 4090](https://gpuperhour.com/providers/tensordock/rtx-4090); [Thunder: TensorDock alternatives](https://www.thundercompute.com/blog/tensordock-alternatives) (biased: a competitor)
- Salad — [GetDeploying RTX 4090](https://getdeploying.com/gpus/nvidia-rtx-4090); [Salad SCE quickstart](https://docs.salad.com/products/sce/getting-started/quickstart)
- Hyperbolic — [Hyperbolic marketplace pricing docs](https://docs.hyperbolic.xyz/docs/gpu-marketplace-pricing); [usagepricing: Hyperbolic rates raised 2026-07-21](https://www.usagepricing.com/blueprint/activity/hyperbolic-2026-07-21-gpu-rates-raised); [usagepricing: H100 increase 2026-08-04](https://www.usagepricing.com/blueprint/activity/hyperbolic-2026-08-04-h100-rate-increase)
- DigitalOcean — [DigitalOcean GPU Droplet pricing](https://www.digitalocean.com/pricing/gpu-droplets)
- Jarvislabs — [Jarvislabs pricing](https://jarvislabs.ai/pricing)
- Prime Intellect — [Product Hunt](https://www.producthunt.com/products/prime-intellect); [Prime Intellect competitor page (Sep 2024)](https://www.primeintellect.ai/competitor); [apis.io plans](https://apis.io/plans/prime-intellect/prime-intellect-plans-pricing/)
- Shadeform — [Shadeform](https://shadeform.com/)
- CoreWeave — [deploybase CoreWeave review](https://deploybase.ai/articles/coreweave-review); [Thunder: CoreWeave alternatives](https://www.thundercompute.com/blog/coreweave-alternatives)

Market context:
- GetDeploying's cross-provider table today (2026-10-31) shows these cheapest / median on-demand prices:
  - RTX 3090: $0.10 / $0.20
  - RTX 4090: $0.27 / $0.53
  - RTX 5090: $0.35 / $0.72
  - L4: $0.42 / $0.88
  - A10: $1.00 / $1.55
  - A100 80 GB: $0.89 / $1.83
  - H100: $1.80 / $3.99

  Several of the cheapest entries come from small providers (HyperAI, Theta EdgeCloud, Packet.ai) that are outside this list. — [GetDeploying cloud GPU reference](https://getdeploying.com/reference/cloud-gpu)
- GetDeploying reported a median 4090 price of $0.49 across 15 providers on 2026-10-17. The cheapest in-stock 4090 was Salad at $0.33. — via [GetDeploying RTX 4090](https://getdeploying.com/reference/cloud-gpu/nvidia-rtx-4090) (search snippet)

### Inferences
- For a small CNN that ran 45 min–4 h on an M1, a 3090/4090/A6000-class card will probably give most of the speedup. An A100/H100 costs 3–10x more and likely won't buy proportionally more speed unless the data loader and batch size are tuned for it. The cost of one run then becomes cents, not dollars. Example: a 30-min run on a $0.34 4090 is about $0.17; on a Secure $0.89 4090 it is about $0.45.
- The idle-pod cost that sank the earlier RunPod setup is avoidable. Keep the 1.1 GB dataset on a **network volume** (about $0.08/month) and create and terminate a pod per experiment. Don't stop a pod with a volume disk: that costs $0.20/GB/mo, and the GPU may be gone when you restart.
- DigitalOcean bills powered-off droplets in full, so "stop between runs" saves nothing there. Only destroy-and-recreate works.
- Thunder Compute's model fits this workload: per-minute billing, disk billed only while running, $0 egress, a first-class CLI, and the cheapest 48 GB card ($0.35 A6000). Its prices look higher than earlier in 2026 (the A100 is now $1.09), so check again.

### Gaps
- No official October 2026 Vast.ai per-GPU prices were found; the Vast price grid isn't in the page text. The quoted figures come from trackers (Aug 2026) and may be stale.
- Lambda's billing granularity and filesystem storage price aren't on its pricing page.
- Paperspace (now part of DigitalOcean): I didn't find a current Paperspace-specific price list. SkyPilot docs still list Paperspace as a separate provider.
- Prime Intellect and Hyperbolic: no current official per-GPU price sheet was found. The Hyperbolic pricing page returned 404.
- Shadeform markup and billing granularity are not stated.

## Scriptability from bash (create → rsync/ssh → run → fetch → destroy)

### Takeaway
RunPod, Vast.ai, Thunder Compute, Lambda, DigitalOcean, Jarvislabs and Shadeform all have an official CLI and/or REST API, so the whole cycle can be scripted. SkyPilot wraps most of them behind one YAML and `sky launch` / `sky down`. Salad is container-group/portal oriented with no SSH documented, so it fits a script-driven rsync workflow poorly.

### Cited Findings
- Vast.ai links a CLI quickstart, CLI docs, a Python SDK and an API reference. — [Vast.ai pricing](https://vast.ai/pricing)
- Vast has `stop-instance` and `start-instance` CLI commands. — [Vast CLI stop-instance](https://docs.vast.ai/cli/reference/stop-instance)
- Thunder Compute: `tnr create` CLI, REST API, MCP server, VS Code extension. — [Thunder Compute pricing](https://www.thundercompute.com/pricing)
- Shadeform: CLI and MCP server, one console across 30+ providers. — [Shadeform](https://shadeform.com/)
- Jarvislabs lists a CLI. — [Jarvislabs pricing](https://jarvislabs.ai/pricing)
- Salad SCE: deploy a container image with replicas through the Portal; shell commands and logs come from Instance Details; storage is ephemeral; no SSH is described. — [Salad SCE quickstart](https://docs.salad.com/products/sce/getting-started/quickstart)
- SkyPilot supports:
  - the big clouds: AWS, GCP, Azure, OCI
  - Kubernetes and Slurm
  - GPU clouds: Lambda, RunPod, Paperspace, Vast.ai, Fluidstack, Cudo, Shadeform, Nebius, Seeweb (EU), Verda (EU), Prime Intellect, Together AI, CoreWeave

  Most of the GPU-cloud integrations are "community-maintained". **Not listed:** DigitalOcean GPU Droplets, Hyperbolic, TensorDock, Thunder Compute, Salad. — [SkyPilot installation docs](https://docs.skypilot.ai/en/latest/getting-started/installation.html)
- RunPod Serverless bills from worker start until it fully stops, rounded up to the second. Pulling the image and downloading cached models while scaled down are not billed. FlashBoot and model caching reduce cold starts. — [RunPod serverless pricing docs](https://docs.runpod.io/serverless/pricing)

### Inferences
- The "one git writer" rule fits a stateless pattern. The laptop rsyncs (or `git archive`s) the current `model/` tree to a fresh instance, runs training, and rsyncs `runs/<exp>/` back, then destroys the instance. The remote never holds a git checkout, which removes the earlier two-writers mess structurally.
- SkyPilot (`sky launch --down` / autostop, with `file_mounts` / `workdir` sync) can choose the cheapest of RunPod/Vast/Lambda/etc. per run. Its integrations for the neoclouds are community-maintained, so expect occasional breakage.
- RunPod Serverless could also work: a Docker image with PyTorch, the dataset on a network volume, and code sent as the job payload. That avoids SSH entirely, at about 2–4x the pod price per GPU-hour. For a few runs a day that is still cents to low dollars.

### Gaps
- SkyPilot's managed-jobs/autostop features weren't on the page fetched. My understanding that `sky launch --down` and `autostop` exist comes from prior knowledge and is not cited here.
- Lambda and DigitalOcean CLI specifics weren't verified this session.

## Cold start, availability, reliability, host variance, trust

### Takeaway
Availability is the main operational risk on marketplaces. RunPod's cheapest Community 4090 was out of stock today. Vast.ai restarts can hang in "scheduling" because the GPU got re-rented, and its storage is pinned to one host. TensorDock has reported empty availability. Hyperbolic dropped consumer cards. A pattern that creates a fresh instance on any available host with the data on portable storage tolerates this best.

### Cited Findings
- RunPod Community 1x 4090 ($0.34) was out of stock and Secure 1x 4090 ($0.89) in stock, checked 2026-10-31. — [GetDeploying RTX 4090](https://getdeploying.com/gpus/nvidia-rtx-4090)
- Vast.ai:
  - When you stop an instance, its GPUs may be reassigned.
  - If it sits in scheduling for more than 30 s, the GPU is likely rented by someone else; the docs suggest creating a new instance.
  - Expired instances can't restart and may be deleted after 48 h.

  — [Vast troubleshooting docs](https://docs.vast.ai/guides/reference/troubleshooting); [Vast manage instances](https://docs.vast.ai/guides/instances/manage-instances)
- Trustpilot reviews from 2026 describe Vast instances stuck in scheduling indefinitely. Vast replied that this is by design for a marketplace. — [Trustpilot Vast.ai](https://no.trustpilot.com/review/vast.ai)
- Vast volumes are tied to one physical machine and can only be reattached on that host. — [Vast.ai storage docs](https://docs.vast.ai/documentation/instances/storage/types)
- Community comments (via Reddit mirrors, dates unconfirmed): Vast.ai has "more variability and occasional reliability headaches"; it is "not reliable for production inference, but for training, they are good". — [Vast.ai review 2026 / search summary](https://www.fast.io/resources/vast-ai-review-2026.md); [r/deeplearning mirror](https://lr.in.psf.lt/r/deeplearning/comments/1kmuf39/where_do_you_get_your_gpus/msd9x6c/?context=3)
- TensorDock: a competitor says availability has been widely reported as empty or unresponsive since late 2025 (biased source). Another source says it stabilized after acquisition by Voltage Park. — [Thunder: TensorDock alternatives](https://www.thundercompute.com/blog/tensordock-alternatives); [Spheron TensorDock 2026](https://www.spheron.network/blog/tensordock-pricing-2026/)
- Salad runs on consumer machines; instances can be reallocated and you aren't billed while down. — [Salad SCE quickstart](https://docs.salad.com/products/sce/getting-started/quickstart)
- RunPod Serverless cold-start figures conflict: one third-party source says about 500 ms for flex, another 30–60 s billed. The official docs only say image pull isn't billed and FlashBoot reduces it. — [RunPod serverless pricing docs](https://docs.runpod.io/serverless/pricing); [hivenet RunPod pricing guide](https://www.hivenet.com/post/runpod-pricing-complete-guide-to-gpu-cloud-costs)

### Inferences
- Trust and security: on Vast community hosts and Salad, the code and the (open-licensed) DOP/TrueDOP data run on third-party machines. For this project the data is dl-de/zero open imagery and the code is meant to be open-source, so the confidentiality risk is low. The bigger risk is a flaky host wasting a run. Mitigations: filter Vast offers by reliability score and datacenter flag, and use a per-run wall-clock timeout in loop.sh.
- If the dataset can't be pinned to a Vast host, plan on re-uploading 1.1 GB per run, or pulling it from object storage or a public URL. The TrueDOP sources are open WMS, but the processed rasters live locally. A fast host downlink makes that tens of seconds.

### Gaps
- No measured cold-start times (instance create → SSH ready) were found for any provider. They should be measured directly.

## Free credits / signup bonuses

### Takeaway
Signup credits are small (about $5–15) and mostly reported by third parties. Startup programs (RunPod, Vast) are larger but target companies, not individuals.

### Cited Findings
- RunPod: signup credit about $10, per most third-party guides. A RunPod-hosted page advertises a random $5–500 bonus (a promotion, unverified). The startup program has a $1,000 starter tier, or $25k bonus on a $50k load (verified Sep 2026). — [search summary: perkstack free GPU credits, updated 2026-10-17](https://perkstack.co/blog/free-gpu-credits-startups); [guptadeepak RunPod startups](https://guptadeepak.com/startup-offers/programs/runpod-startups)
- Lambda: $10–15 signup credit per guides (least certain). — [Spheron free GPU credits 2026](https://www.spheron.network/blog/free-gpu-cloud-credits-2026/)
- Vast.ai: about $5 signup credit (varies). The startup program (announced 2026-02-06) gives $2,500 in credits. — [Vast.ai Startup Program](https://vast.ai/article/vast-ai-startup-program)

### Inferences
- Credits don't decide anything here. At $0.2–0.9/h and a few runs a day, monthly spend is likely single-digit to low-double-digit dollars.

### Gaps
- No credit info was found for Thunder Compute, TensorDock, Salad, Jarvislabs, Prime Intellect or Shadeform.

## Payment and EU/Germany constraints

### Takeaway
Most marketplaces are prepaid: Vast is explicitly prepaid and accepts crypto; Jarvislabs bills in USD outside India. Prices exclude VAT; Lambda states "excluding applicable sales tax/VAT". No provider found refuses German individuals, but minimum top-ups weren't documented in the sources found.

### Cited Findings
- Vast.ai requires prepaid credits. Crypto via Crypto.com and Coinbase is supported. Instances stop (not destroy) when the balance hits zero. — [Vast billing docs](https://docs.vast.ai/billing)
- Lambda prices exclude applicable sales tax/VAT/GST. — [Lambda pricing](https://lambda.ai/pricing)
- RunPod has a default spend limit of $80/h. — [RunPod serverless pricing docs](https://docs.runpod.io/serverless/pricing)
- Jarvislabs bills in USD for customers outside India. — [Jarvislabs pricing](https://jarvislabs.ai/pricing)
- EU-based GPU clouds supported by SkyPilot: Seeweb (Italy) and Verda (formerly DataCrunch, Finland). Nebius is EU-founded. — [SkyPilot installation docs](https://docs.skypilot.ai/en/latest/getting-started/installation.html)

### Inferences
- For a German individual, expect 19% German VAT on top of the listed USD prices from providers that charge EU VAT (reverse charge applies only to businesses with a VAT ID).
- Verda and Seeweb are worth a look if EU data residency or EUR invoicing matters, but they weren't priced here.

### Gaps
- Minimum top-ups for RunPod, Vast, Thunder and TensorDock weren't found in this session.
- No explicit statement was found on whether each provider charges German VAT.

## CoreWeave — why not suitable

### Takeaway
CoreWeave is built for teams running Kubernetes/Slurm clusters with multi-GPU and multi-day commitments. It isn't meant for a solo user running occasional 1-GPU jobs.

### Cited Findings
- HGX H100/A100 instances are sold in 8-GPU blocks; GH200 is the single-GPU exception. There is no self-serve signup for that range. — [Thunder: CoreWeave alternatives](https://www.thundercompute.com/blog/coreweave-alternatives) (competitor, biased)
- No single on-demand A100; the minimum is 8x A100. — [deploybase CoreWeave vs Paperspace](https://deploybase.ai/articles/coreweave-vs-paperspace)
- Minimum commitments are typically one week, and there is no simple "spin up GPU" button. — [deploybase CoreWeave review](https://deploybase.ai/articles/coreweave-review)
- Conflicting third-party listing: per-second billing, no minimum, self-serve API; but minimum spend in the hundreds of dollars a month. — [aitoolsatlas CoreWeave](https://aitoolsatlas.ai/tools/coreweave/pricing)
- SkyPilot reaches CoreWeave only through its Kubernetes (CKS) and Slurm (SUNK) offerings. — [SkyPilot installation docs](https://docs.skypilot.ai/en/latest/getting-started/installation.html)

### Inferences
- Even if self-serve exists, the interface (Kubernetes) and the minimum spend are out of proportion to a few small CNN runs a day.

### Gaps
- No official CoreWeave page was fetched; all claims are third-party and partly conflicting.
