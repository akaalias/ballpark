# The Library — human inspiration notes

Maintained by the human researcher. These notes do not fix your answer and
you are free to ignore them — pick one up only when it genuinely fits your
read of the experiment history. If you build on an entry, say so in your
hypothesis. Entries are inspiration, not instructions; you own the design.

---

## L1 · Canonicalize the night: denoise toward the clean render (2026-07-21)

> **NOT APPLICABLE on the berlin-slim branch.** This entry depends on the
> frozen relighting sim for supervised clean/night pairs, and relighting is
> disabled here — training and eval both use the raw daytime imagery as
> fetched, so there is no night frame to canonicalize and no pair to learn
> from. Kept for the main branch; do not build on it on this branch.

The worst-case input is the noisy night frame. What if the hard part —
seeing through gain noise and washed-out color — were handled explicitly,
so the localizer works on something closer to a clean, high-contrast
rendering of the ground?

Unfair advantage of this codebase: the frozen relighting sim provides
**pixel-aligned supervised pairs for free** — for any training location you
can load both the clean daytime reference crop and its rendered night
frame (same cx/cy through `pipeline.dataset`, different lighting bucket, or
the reference image itself). No labeling cost, unlimited pairs.

Two flavors, different deployment economics:

- **A. Explicit pre-model**: a small translator (tiny U-Net-ish) runs
  first; the localizer sees the cleaned image. It flies, so it spends the
  size / 250 ms latency budget — keep it small.
- **B. Training-only reconstruction auxiliary** (the budget-free version):
  keep one deployed model, but during training attach a decoder that must
  reconstruct the clean daytime render from the encoder's features, then
  discard it. Forces the encoder to internally canonicalize lighting and
  noise at zero inference cost — lives entirely in the "never flies" lane.

Prior art the idea rhymes with: night→day translation for place
recognition (ToDayGAN-family), denoising pretraining, reconstruction
auxiliaries for robust features.

---

## L2 · World-model ideas for the held-out photograph (2026-10-30)

A deep-research pass over JEPA / I-JEPA / V-JEPA 2 / DINO-WM / Dreamer /
TD-MPC / MuZero / SPR and the remote-sensing and place-recognition work that
borrows from them, asked one question: what in that literature helps a
0.8M-parameter CNN locate Berlin from a photograph it never trained on? Full
report with every citation: `reports/JEPA world model ideas for Berlin.md`;
the five source notes are in `research_notes/JEPA world model ideas for
Berlin/`. Read the report's "Ranked shortlist" section before designing.

The conclusion is a *training-only discipline*, not an architecture: predict
representations of related views, never pixels, against a target that does
not move (stop-gradient or EMA), through a normalised latent that cannot
collapse. Every expensive part (teacher, predictor, projector, VICReg terms)
is scaffolding dropped before export, so the size gate never sees it. The
harness holds the richest "related view" any of those papers had: three
photographs of the same pixel-aligned ground with every crop's position,
heading and survey known.

Ranked, each one focused change (evidence level in the report):
1. **Cross-survey latent consistency** — cosine between projected trunk
   features of the same position in survey A and `sg(survey B)`, VICReg
   variance/covariance guard. The top rung of SSL4EO's measured ladder.
2. **Broad photometric + geometric augmentation** (brightness, contrast,
   gamma, hue/saturation, blur, noise, ±12 px jitter). Cheap; fixes the
   re-exposure part of the gap, not the season part.
3. **Cached DINOv2-S patch-feature distillation** with a cross-survey term
   (student on survey A matched to teacher on survey B). Apache-2.0 teacher;
   one ~13 min cache pass per survey on the M1, then free per experiment.
   Largest documented effect in the scene-coordinate literature.
4. **Translation/rotation as an action** — a 2-layer transition head
   predicts the neighbour crop's latent from (latent, sin-cos dx,dy,dθ).
   The world-model lineage's genuine contribution; equivariance without an
   equivariance loss.
5. Cross-survey masked latent prediction (CNN-JEPA form) — highest risk at
   this scale; its value is partly delivered by 4.
6. Mapping/query split for confidence calibration.  7. Distance-aware
   (metres) soft targets.  8. SeCo-style survey-identity side head.
   9. CACo change gate on the consistency term.

What does NOT transfer: anything needing a sequence at inference (RSSM,
imagination, MCTS, latent maps), pixel generators (Genie, Cosmos), latent
action discovery (actions are known), from-scratch SSL on a 0.8M CNN inside
an hour, un-normalised L2 latent losses (collapse), decode-time consensus
(refuted in-project), learned aleatoric confidence (generalises worst under
map change). Expect the held-out survey to stay below the training-survey
diagnostics even after the fix; a narrowing gap is the evidence.
