# Model Selection

Results of the architecture sweep and tie-break experiment used to select the final SkiNet encoder–merge combination.

## Figure 1 — Paired slope graph

Each line connects the val Dice score of one seed run under the two tied configurations
(classical/He2 vs classical/attention-gate). The slope direction shows which configuration
performed better for that seed.

```{image} _static/model_selection/E2_fig1_paired_slopegraph.png
:alt: Paired slope graph comparing classical/He2 and classical/attention-gate val Dice across 10 seeds
:width: 80%
:align: center
```

*Figure 1. Per-seed val Dice for the two top-ranked encoder–merge configurations across the 10-seed tie-break. The paired t-test (p = 0.18, mean diff +0.0016) was non-significant; the rank-1 result (classical encoder + He2 merge) was adopted by the conservative selection rule.*

---

## Figure 2 — Forest plot of paired differences

Each row shows the Dice difference (classical/He2 − classical/attention-gate) for one seed,
with 95 % confidence intervals. The pooled estimate and its interval are shown at the bottom.

```{image} _static/model_selection/E2_fig2_forest_paired_diff.png
:alt: Forest plot of per-seed Dice differences between classical/He2 and classical/attention-gate
:width: 80%
:align: center
```

*Figure 2. Forest plot of paired Dice differences (He2 − attention-gate) across 10 seeds. The pooled mean difference is +0.0016 (95 % CI crosses zero), confirming the non-significant result and supporting selection on full-grid rank rather than tie-break magnitude.*
