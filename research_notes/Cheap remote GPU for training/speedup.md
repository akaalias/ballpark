# Speed-up of small-CNN PyTorch training on cloud GPUs vs. an Apple M1 (MPS), and what limits it

Scope note: published benchmarks almost never cover *this* regime (a ~0.7 M-param CNN, batch 64, heavy per-step Python/augmentation). The most informative evidence turned out to be (a) the project's own timings and code, and (b) general findings on overhead-bound GPU training. Local measurements below are marked **[local]**; they were taken on this machine (`MacBookAir10,1` = **fanless M1 MacBook Air**, 8-core GPU) on 2026-10-31 while experiment 6 was training, so CPU timings are slightly pessimistic.

## 1. Published benchmarks: Apple M-series (MPS/Metal) vs NVIDIA GPUs

### Takeaway
On large, compute-bound CNNs (VGG16, ResNet-50) an M1 Max is roughly 8x slower than an RTX 3090, and a base M1 has about one third of an M1 Max's GPU. So the compute-bound ratio is roughly 20–30x for a 3090 and roughly 30–50x for a 4090 or A100 relative to a base M1. No published benchmark measures the small-model, small-batch regime this project runs in. There the ratio is set by overhead, not FLOPs, and is much smaller (see §2 and §4).

### Cited Findings
- tlkh's tf-metal-experiments (TensorFlow-Metal, not PyTorch): M1 Max ≈ 140 img/s ResNet-50 training at batch 128, ~42 W GPU power. RTX 3090 reference ≈ 1,100–1,185 img/s, but run with XLA + FP16, so the comparison is not like-for-like. The author found that a larger batch did not raise M1 Max throughput — [tlkh/tf-metal-experiments](https://github.com/tlkh/tf-metal-experiments)
- M1 Pro (14-core GPU): 57.9 img/s ResNet-50 at batch 128, about half of the M1 Max — [tlkh issue #13](https://github.com/tlkh/tf-metal-experiments/issues/13); a forum user reports ~65 img/s on an M1 Pro — [MacRumors thread](https://forums.macrumors.com/threads/apple-silicon-deep-learning-performance.2319673)
- Community summary: "M1 Max is around 8 times slower than an RTX 3090 (fp16) but consumes 8 times less power" — [MacRumors thread](https://forums.macrumors.com/threads/apple-silicon-deep-learning-performance.2319673)
- Sebastian Raschka, PyTorch MPS, VGG16 on 224x224 CIFAR-10: the M1 Pro GPU trains 3.5x faster than the M1 Pro CPU. Batch 64 was ~20% faster than batch 32 on MPS. His conclusion: MPS is "not a total game changer… we probably still want to stick to conventional GPUs" for training. (His NVIDIA rows, 2080Ti/3080/A100, are in images; the exact ratios could not be extracted as text.) — [Raschka 2022](https://sebastianraschka.com/blog/2022/pytorch-m1-gpu.html)
- Theoretical FP32 peaks: M1 8-core GPU ≈ 2.6 TFLOPS at ~10 W — [gpu-monkey](https://www.gpu-monkey.com/en/benchmark-apple_m1_8_core_gpu-fp32), [Notebookcheck](https://www.notebookcheck.net/M1-8-Core-GPU-vs-M1-Max-24-Core-GPU_10552_10967.247598.0.html). RTX 4090 ≈ 82.6 TFLOPS FP32 — [Notebookcheck](https://www.notebookcheck.net/M2-Pro-16-Core-GPU-vs-NVIDIA-GeForce-RTX-4090-vs-M2-8-Core-GPU_11570_11436_11375.247598.0.html). That is a ~32x paper gap.
- RTX 4090 vs RTX 3090: 1.3–1.9x (TF32) and 1.3–1.8x (FP16) higher training throughput, depending on model — [Lambda](https://lambda.ai/blog/nvidia-rtx-4090-vs-rtx-3090-deep-learning-benchmark)
- A100 vs V100 on PyTorch convnets: 2.2x (FP32) and 1.6x (mixed precision) — [Lambda](https://lambda.ai/blog/nvidia-a100-vs-v100-benchmarks)
- A10G vs L4 vs T4: no measured PyTorch training head-to-head found. Spec pages list A10G at 600 GB/s vs L4 at 300 GB/s memory bandwidth, and call the T4 and L4 inference-oriented — [gpu-markets A10G vs L4](https://gpu-markets.pages.dev/compare/a10g-vs-l4), [arXiv 2604.14552 (T4/L4 inference)](https://arxiv.org/pdf/2604.14552)
- Single-GPU 4090 vs A100 training for small CNNs: no reliable benchmark found. Trooper.ai's pages are inference/diffusion only, mirrors disagree (4090 wins 15/26, or A100 +14%), and no methodology is published — [trooper.ai](https://www.trooper.ai/benchmarks-compare-RTX-4090-with-A100)

### Inferences
- Compute-bound ratio, base M1 → NVIDIA: about 20–30x for a 3090 (8x M1 Max, times ~3 because the M1 has 8 GPU cores vs 24–32). A 4090 adds ~1.5x on top of that, and an A100 sits in a similar class to the 4090 for FP32/TF32 work. These are upper bounds that only a compute-saturating workload reaches.
- On paper, T4 < L4 < A10G < 3090 < 4090 ≈ A100 < H100 for a single small job. For a 0.7 M-param net none of the top three will be saturated (§2).

### Gaps
- No PyTorch-MPS vs CUDA benchmark found for ResNet-18/MobileNet-class nets at small batch. No base-M1 vs 4090/L4/A100 measurement in PyTorch.
- The L4 FP32 spec (often quoted ~30 TFLOPS) was not verified from a primary source.
- No measured T4/L4/A10G *training* comparison.

## 2. When a big GPU is wasted: overhead-bound small models; batch size, compile, AMP, channels_last, workers

### Takeaway
A small CNN at batch 64 is limited by Python dispatch, kernel launch latency, host↔device syncs and CPU-side data preparation, not by GPU FLOPs. In that regime an A100/H100 runs no faster than a 4090 or even an L4, because both wait on the same CPU. Big GPUs only pay once the data pipeline lives on the GPU, the batch grows, and launch overhead is fused away (torch.compile / CUDA graphs).

### Cited Findings
- Small models are "overhead-bound": Python, dispatcher and kernel-launch latency dominate when kernels are small, leaving gaps on the GPU timeline. `.cpu()`/`.item()` in the loop forces a sync and stalls the pipeline. Remedies: torch.compile (`reduce-overhead` uses CUDA graphs), CUDA graphs directly, fewer syncs, profile first. The author saw an A100 training loop at ~15% utilisation, and in that example a larger batch did *not* help because the cost was per-op overhead — [dev.to, Alan West](https://dev.to/alanwest/why-your-pytorch-training-crawls-on-a-beefy-gpu-and-how-to-fix-it-52g8)
- NVIDIA: workloads with low GPU utilisation and many small kernel launches are the strongest candidates for CUDA graphs, and can approach 100% utilisation. `torch.compile(mode="reduce-overhead")` is the easiest route, but graph breaks fragment it. Graphs require static shapes and control flow — [NVIDIA CUDA Graph best practices](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/best-practices.html), [NVIDIA troubleshooting](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/performance-issues.html)
- If data loading exceeds model step time, that overhead shows up in every iteration, and GPU-side optimisations don't help — [PyTorch forum](https://discuss.pytorch.org/t/pytorch-is-much-slower-than-tensorflow/174519)
- An existence proof of what a small CNN can do on one GPU when the whole pipeline is GPU-resident: airbench trains to 94% on CIFAR-10 in 3.83 s (3.29 s with torch.compile) on one A100. The data is held on the GPU and augmentation is done there — [Jordan 2024, arXiv 2404.00498](https://arxiv.org/html/2404.00498v2)
- Deferring `.item()` gave only ~3% on CUDA for a small MLP (one benchmark), so sync removal alone is not a large win on CUDA — [HackerNoon](https://hackernoon.com/your-pytorch-model-is-slower-than-you-think-this-is-the-reason-why?source=rss)

### Inferences
- Batch size: going from 64 to 256–512 is the single largest lever for GPU utilisation with a net this small. It also changes the optimisation (fewer steps per epoch, so the LR may need scaling or warm-up). That makes it a research change, not just an infra change, and it should be validated against the batch-64 baseline score.
- AMP/bf16 and channels_last help compute-bound convs on Ampere/Ada tensor cores. They are second-order here until overhead is removed. bf16 is supported on 3090/4090/L4/A10G/A100/H100 but not on T4 (FP16 only).
- num_workers is irrelevant to the current train.py: it does not use a DataLoader, and crops are produced synchronously in the main process (§4).

### Gaps
- No controlled measurement of torch.compile/AMP/channels_last gains for a ~0.7 M-param CNN on a 4090 vs A100. Expect them to be workload-specific.

## 3. MPS-specific slowness and M1 thermals

### Takeaway
MPS pays heavily for every host sync, scales worse at small batch than CUDA, and has seen sporadic memory-leak and fallback issues. This machine is a **fanless MacBook Air**, which throttles under sustained GPU load, but no measured percentage exists for M1 Air training.

### Cited Findings
- On MPS, most step time in a reported slow loop came from `loss.cpu()…`, which forces a GPU sync. Fix: accumulate on-device and sync once per epoch — [PyTorch forum: Dataloader slows down with MPS](https://discuss.pytorch.org/t/dataloader-slows-down-when-training-with-mac-mps/196185)
- A low-authority blog claims MPS needs larger batches than CUDA to pay off, and that `.item()`, printing and logging force the GPU to drain its queue — [macmyths](https://macmyths.com/pytorch-mps-slower-than-cpu) (unverified)
- An MPS memory leak during training was reported in mid-2025 — [PyTorch weekly GitHub report](https://buttondown.com/weekly-project-news/archive/weekly-github-report-for-pytorch-may-26-2025-june-2906/)
- Fanless Air throttling: anecdotal reports of slowdown after ~5 min of load (M2 Air, Reddit 2022) — [r/deeplearning](https://redlib.hbubli.cc/r/deeplearning/comments/w8s13f/anyone_has_experience_with_doing_deep_learning). Diagnose with `powermetrics`: the signature is GPU power settling below its early peak while die temperature sits at a flat ceiling — [llmconfigurator guide](https://llmconfigurator.com/en/guides/troubleshooting/macbook-thermal-throttling-llm). A claimed 40–60% throughput loss on a fanless M4 Air is unverified — [bit-axon page](https://skyoo2003.github.io/bit-axon/papers/thermal-training/)
- Fan-cooled M1 Max is reported to stay "cool as a cucumber" during long training — [tlkh issue #4](https://github.com/tlkh/tf-metal-experiments/issues/4)
- **[local]** `sysctl hw.model` = `MacBookAir10,1`, i.e. a fanless M1 Air.

### Inferences
- With ~8 sync points per step in train.py (§4), MPS is close to worst-case: each step drains the queue repeatedly, so GPU work never overlaps the next step's CPU work.
- Thermal throttling probably costs some fraction of sustained throughput on this Air. That share of the remote speed-up is free, but its size is unmeasured. It can be checked with `sudo powermetrics --samplers gpu_power` during an epoch.

### Gaps
- No source found on which specific ops in this code, if any, fall back to CPU on MPS (e.g. reflect-pad, grouped conv with groups = B·C=192). PyTorch's MPS fallback only activates with `PYTORCH_ENABLE_MPS_FALLBACK=1`, otherwise it errors, so the run completing suggests no silent fallback. Not verified.
- No measured throttling percentage for an M1 Air under PyTorch training.

## 4. The actual train.py: likely bottleneck, cheap fixes, expected remote wall-clock

### Takeaway
The current loop runs the M1 GPU at only ~3–4% of its paper FLOPs. Time goes to (a) single-threaded CPU crop rotation via PIL, serialised with the GPU, and (b) ~200+ tiny kernels and ~8 host syncs per 64-view step. Moved to a 4090 **unchanged**, expect ~3–5x (≈45–80 min for a 4 h M1 job), and an A100/H100 would be no faster. With a GPU-resident pipeline (rasters on GPU, rotation via `grid_sample`, batch 256–512, no per-step syncs, bf16, compile), the GPU time per experiment probably drops to a few minutes (20–50x+). Pod start-up, data upload and the ~6 min of local agent work then dominate wall-clock.

### Cited Findings ([local] = this repo/machine; sources for general claims are in §2)
- **[local] Workload.** Model `CellLocNet`: four stride-2 3x3 conv+BN+ReLU layers (3→32→64→128→160), GAP, then a 160→2,970 cell head plus a 2-d offset head. Input 128x128, batch 64, Adam, cosine LR (`model/model.py`, `model/train.py`). By my count it is ~54 M MACs per view forward, ~0.3 GFLOP per view for training, ~0.7 M params (ONNX ~3 MB fp32 matches).
- **[local] Data.** 72,712 lattice positions × 3 training captures = 218,136 views per epoch, 8 epochs, about 3,400 steps per epoch. Rasters are 6,939 × 6,828 RGB, i.e. ~142 MB per capture as uint8 and ~426 MB for the three training captures. The training PNGs total ~308 MB, so the 1.1 GB area folder is mostly other files and none of it needs to be shipped beyond those three PNGs plus meta.
- **[local] Measured M1 wall-clock.** `train_seconds` = 3,500 / 4,079 / 6,157 / 6,142 s for capture-holdout experiments 2–5. Experiment 5 averages ~770 s per epoch, about 0.23 s per step, about 280 views/s. At 0.3 GFLOP per view that is ≈ 90 GFLOP/s, roughly 3–4% of the M1's 2.6 TFLOPS paper peak. The run is not compute-bound. Experiment 6 (running now) is slower, at about 20+ min per epoch.
- **[local] CPU crop extraction.** `extract_crop` takes a 182 px window and runs `PIL.Image.rotate` (bilinear) once per view, single-threaded in the main process. Measured at **0.55 ms per crop**, about 120 s per epoch (~16% of an epoch), fully serialised with the GPU because there is no DataLoader, worker or prefetch.
- **[local] Host syncs per step.** `loss.item()`, `hit…cpu()`, `mixed.cpu()`, several `int(...)` reductions, `float(conf…sum())`, and `bool(mixed.any())` / `float(mmw…)` inside `cross_survey_blend`. That is about 8 GPU→CPU syncs per step.
- **[local] Kernel count.** `augment_views` (~30 elementwise and grouped-conv ops), `cross_survey_blend` (~15), `make_soft_targets` (a dense [64, 2970] target built by 4 scatters), `residual_offset_target` (~15), the model fwd/bwd, and Adam. On the order of a few hundred tiny kernels per step: the overhead-bound pattern from §2.
- **[local] Device pick.** `device = "mps" if available else "cuda"…`, so the file runs on a CUDA box unchanged.

### Inferences (estimates, with uncertainty)
- **Where the M1 time goes (inferred):** about 15% CPU crop rotation and about 85% the GPU step, which is itself mostly launch/dispatch overhead and sync stalls on MPS, plus any throttling.
- **Unchanged code on a cloud GPU.** A CUDA step of this shape should take ~5–15 ms (launch-bound), so ~20–50 s of GPU per epoch. Crop rotation stays at ~110–220 s per epoch: cloud x86 or Graviton vCPUs are typically not faster per core than an M1 Firestorm core, so that part does not speed up. Per epoch, then, ~150–270 s against ~770 s on M1.
  - Speed-up **~3–5x** (low confidence: ±50%).
  - A 4 h M1 job becomes ~50–80 min; experiments 4/5 (1.7 h) become ~20–35 min.
  - **4090 ≈ L4 ≈ A10G ≈ A100 ≈ H100 here**, because the floor is single-threaded CPU plus launch overhead. A T4 may be a bit slower. Buying an A100 for this code is wasted money.
- **Cheap code changes, in priority order** (all training-only, the exported ONNX graph is unchanged; they also help on the M1):
  1. **Upload the three training rasters once as uint8 tensors to the GPU** (~426 MB). Replace per-view PIL rotate with batched `F.affine_grid` + `F.grid_sample` (bilinear) on index-gathered windows. This removes the CPU floor and the CPU→GPU copies. Caveat: GPU bilinear rotation is not bit-identical to PIL's. It is fine for training, but it is a semantic change the loop should log.
  2. **Remove per-step syncs.** Accumulate loss, hits and confidence as device tensors and `.item()` once per epoch. Drop the `bool(mixed.any())` / `float(mmw)` branches.
  3. **Batch 256–512**, with LR scaling or warm-up. This needs a score check, because it changes optimisation.
  4. **`torch.autocast(dtype=torch.bfloat16)` + `channels_last`**, on Ampere/Ada/Hopper; on a T4 use fp16 + GradScaler.
  5. **`torch.compile`** on the model and the augmentation function (`mode="max-autotune-no-cudagraphs"` or `reduce-overhead`; shapes are static once the last partial batch is dropped). This fuses the ~45 elementwise augmentation ops.
  6. Build the dense [B, 2970] soft target more cheaply. Minor.
- **Expected with changes 1–5:** compute is ~218k × 0.3 GFLOP ≈ 65 TFLOP per epoch. At a realistic 15–30% of a 4090's peak (~80 TFLOPS FP32, more in bf16), that is a few seconds of pure math per epoch; the augmentation passes are memory-bound and add maybe as much again. So **~10–40 s per epoch, ~2–6 min for 8 epochs** on a 4090 (order-of-magnitude confidence only), a **~20–50x+** speed-up over the M1 Air.
  - An L4 is roughly 2–3x slower than a 4090 here (half the bandwidth, fewer FLOPs; spec-based, unverified), so ~5–15 min.
  - An A100 is about 4090-class (it has more bandwidth, which helps the memory-bound augmentation), and an H100 is somewhat faster. Neither is worth 4–8x the price at this model size.
- **Wall-clock floor per experiment.** Once training is minutes long, fixed costs dominate:
  - pod start and image pull, plus `pip install torch`: ~2–10 min unless a cached image or persistent pod is used;
  - uploading ~310 MB of PNGs, once per persistent pod;
  - the local design and implementation agents (~6 min per experiment, `timings.json`: design 289 s, implement 51 s);
  - scoring (~22 s).
  
  A persistent pod kept warm across a loop session is what makes the speed-up real.
- **Leakage hygiene (project rule).** Ship only the three `role: train` PNGs and meta to the remote box. Never ship `truedop_2024.png`, so the "nothing under model/ opens the eval capture" contract also holds physically.

### Gaps
- The split between MPS overhead and throttling in the ~650 s per epoch of non-crop time was not profiled. A 2-minute `torch.profiler` run, or timing a fixed 200 steps with crops precomputed, would settle it.
- The cloud-CPU per-crop PIL time was not measured (assumed similar to or slower than the M1's 0.55 ms).
- No benchmark exists for this exact model on any NVIDIA GPU. The first remote run should log steps/s for the unchanged and the optimised code.

## 5. Rough cost per experiment

### Takeaway
A cheap marketplace 4090 is the sweet spot: roughly **$0.10–0.30 per experiment** with unchanged code, **< $0.10** optimised. An L4/A10G on a hyperscaler costs ~2–3x more for the same or slower result. An A100/H100 buys nothing at this model size.

### Cited Findings
- RTX 4090:
  - from $0.14/h on Vast.ai, $0.34/h on RunPod Community, $0.74/h on RunPod Secure (Sept 2026) — [Hostinger price comparison](https://www.hostinger.com/tutorials/?p=158956);
  - another aggregator lists Vast $0.52 / RunPod $0.69 — [Spheron](https://www.spheron.network/blog/runpod-vs-vastai-2026/).
  
  Prices vary daily, and the cheapest offers are often interruptible.
- L4: ~$0.33–0.39/h on marketplaces; GCP G2 L4 from ~$0.71/h — [Hostinger](https://www.hostinger.com/tutorials/?p=158956)
- A10G ~$0.54/h on-demand, T4 ~$0.22/h — [gpu-markets A10G vs T4](https://gpu-markets.pages.dev/compare/a10g-vs-t4)
- A100 80 GB:
  - RunPod PCIe ~$1.39/h, SXM ~$1.59/h;
  - Vast quoted ~$0.67–0.74/h, but often $1.50+ when busy.
  
  — [Hostinger](https://www.hostinger.com/tutorials/?p=158956), [deploybase](https://deploybase.ai/articles/cheapest-gpu-cloud-in-2026-provider-pricing-ranked)

### Inferences
Costs for one 8-epoch experiment equivalent to the M1's ~1.7 h (experiments 4/5), training only, excluding idle pod time:

| GPU, $/h | unchanged code (~25–40 min) | optimised (~3–15 min) | 4 h-M1-class job, unchanged (~50–80 min) |
|---|---|---|---|
| 4090 @ $0.34 | ~$0.15–0.25 | ~$0.02–0.05 | ~$0.30–0.45 |
| L4/A10G @ $0.70–1.00 | ~$0.30–0.65 | ~$0.10–0.25 | ~$0.60–1.30 |
| A100 @ $1.40–2.50 | ~$0.60–1.70 (no faster) | ~$0.10–0.40 | ~$1.20–3.30 |

- Keeping a pod warm for a 12-experiment loop session (~2–3 h of agent and scoring time in between) costs ~$1 on a 4090 vs ~$4–7 on an A100. Idle time, not training, is the dominant cost.
- Net: renting gives a modest 3–5x with zero code changes. The significant win (an order of magnitude or more) needs the GPU-resident data pipeline, and the first three items of that list also speed up the M1. Measure on the M1 after those changes before deciding whether renting is needed at all.

### Gaps
- Live prices were not checked against the providers' own pages (all from aggregators, May–Sept 2026). Egress and storage fees, and Modal/RunPod start-up minimums, were not included.
