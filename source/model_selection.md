# Model Selection

Results of the architecture sweep and tie-break experiment used to select the final SkiNet encoder–merge combination.

## Figure 1 — Paired slope graph

Each line connects the val Dice score of one seed run under the two finalist configurations
(classical/attention-gate (AG) vs classical/He2 (HE2)). The slope direction shows which configuration
performed better for that seed.

```{image} _static/model_selection/E2_fig1_paired_slopegraph.png
:alt: Paired slope graph comparing classical/attention-gate and classical/He2 val Dice across 10 seeds
:width: 80%
:align: center
```

*Figure 1. Per-seed val Dice for the two finalist encoder–merge configurations across the 10-seed tie-break (seeds 100–109). On the pre-registered primary metric — plateau Dice (mean of the last 10 epochs) — AG wins 8 of 10 seeds (0.8300 vs 0.8275); peak Dice is an even split (statistical tie).*

---

## Figure 2 — Forest plot of paired differences

Each row shows the paired plateau-Dice difference Δ = AG − HE2 (classical/attention-gate − classical/He2) for one seed,
with BCa 95 % confidence intervals. The pooled estimate and its interval are shown at the bottom.

```{image} _static/model_selection/E2_fig2_forest_paired_diff.png
:alt: Forest plot of per-seed plateau-Dice differences (attention-gate − He2)
:width: 80%
:align: center
```

*Figure 2. Forest plot of paired plateau-Dice differences (AG − He2) across 10 seeds. The pooled mean difference is +0.0025 (BCa 95 % CI [+0.0006, +0.0048], excludes zero; Wilcoxon p = 0.037, d_z = +0.70), confirming attention-gate as the winner on the pre-registered primary metric. He2's only confirmed advantage is a 13 % throughput edge. → Locked: **classical encoder + attention_gate merge** at lr = 3e-4.*

---

## Full analysis notebook

```{toctree}
:maxdepth: 1

E2-isic2017-unet2d-model-tiebreak-10seed
```
