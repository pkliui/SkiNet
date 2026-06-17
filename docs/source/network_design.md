# Network Design

## Architecture — U-Net 2D (custom, from scratch)

SkiNet is a symmetric encoder–decoder network built entirely from scratch in PyTorch,
without pre-trained backbones. The encoder path progressively compresses spatial
resolution while doubling channel depth at each stage. The decoder path mirrors it,
restoring resolution via transposed convolutions. Skip connections at every resolution
level carry high-frequency spatial detail from encoder to decoder, preventing
boundary information from being lost during downsampling.

## Architecture diagram

:::{figure} _static/arch_overview.svg
:alt: U-Net 2D architecture overview
:width: 100%
<small>Skip connections (not shown) run from each encoder layer to its mirror decoder merge block.
Each upsampling step uses a transposed convolution (stride 2), restoring the spatial dimensions
halved by the corresponding encoder stage.</small>
:::

## Block design

### Encoder block — Classical (selected) (Ronneberger et al., MICCAI 2015)

Below is the encoder used in the final SkiNet (`ClassicalEncoder`). It consists of two sequential
Conv2d(3×3) → BatchNorm → ReLU layers per encoder stage: the first convolution
downsamples (stride 2), the second refines at the reduced resolution (stride 1).
The encoder block was selected in the architecture sweep (see Model selection below), wherethe `he2` and `se`
encoders were also evaluated and appear under Candidate blocks.

```text
h = Conv-BN-Act(x)   # stride 2, downsamples spatial dims
y = Conv-BN-Act(h)   # stride 1, refines at reduced resolution
```

:::{figure} _static/block_encoder.svg
:alt: Classical encoder block data-flow diagram
:width: 100%
<small>Classical encoder block — Conv 3×3 (stride 2) → BN → ReLU → Conv 3×3 (stride 1) → BN → ReLU.</small>
:::

### Decoder block — transposed convolution upsampling

Each decoder stage begins by doubling the spatial resolution with a single
transposed convolution, followed by BatchNorm and ReLU. The kernel size is
set to `encoder_kernel × encoder_stride = 3 × 2 = 6` rather than
the more common 2×2, which eliminates the uneven overlap pattern that
produces checkerboard artefacts in the output mask (Odena et al., 2016).
The stride of 2 exactly inverts the downsampling applied by the corresponding
encoder stage. The output of this block is passed to the attention-gated merge
block below alongside the skip connection.

```text
y = Act(BN(ConvTranspose2d(x)))   # kernel 6×6, stride 2: spatial ×2, channels ÷2
```

:::{figure} _static/block_decoder.svg
:alt: Decoder upsampling block data-flow diagram
:width: 100%
<small>Decoder upsampling block — ConvTranspose2d 6×6 (stride 2, spatial ×2, channels ÷2) → BN → ReLU.</small>
:::



### Merge block — Attention-gated (selected) (Oktay et al., MIDL 2018)

This is the merge used in the final SkiNet (`AttentionGateMerge`). Before merging, the skip connection is
gated by an additive attention gate: the upsampled decoder tensor acts as the gating signal `g`,
and `g` and the skip features `x` are each projected to an intermediate space, summed, then passed
through ReLU → 1×1 conv → BN → Sigmoid to produce a spatial attention map α ∈ [0, 1]. α multiplies
the raw skip features, selectively suppressing irrelevant background activations at the skip
connection.

:::{figure} _static/block_attention_gate.svg
:alt: Additive attention gate data-flow diagram
:width: 100%
<small>Additive attention gate — the decoder gating signal g and skip x are each projected (1×1 conv → BN),
summed, then ReLU → 1×1 conv → BN → Sigmoid yields the spatial map α ∈ [0, 1]. α multiplies the raw
(unprojected) skip x to produce the gated skip features fed into the merge.</small>
:::

The gated skip and the decoder tensor are then each projected through separate 3×3 convolutions and
_summed_ (rather than concatenated). This operation is algebraically equivalent to concatenation and convolution, but
avoids the doubled-channel tensor in memory. The resulting merged sum is followed by two BN→ReLU→Conv refinement convolutions
with an identity shortcut over the merged sum (the two-conv pre-activation "He2" pattern, just with attention gate; He et al., ECCV 2016).

:

```text
attended_skip = AttentionGate(dec, skip)          # α-gated skip features
merged        = conv_x(dec) + conv_skip(attended_skip)
conv1         = Conv(Act(BN(merged)))
conv2         = Conv(Act(BN(conv1)))
output        = conv2 + merged                    # identity shortcut
```

:::{figure} _static/block_attention_gate_merge.svg
:alt: Attention-gate merge block data-flow diagram
:width: 100%
<small>Attention-gate merge — skip is gated by α (W_x/W_g projections → ⊕ → ReLU → ψ → Sigmoid), then
attended_skip and dec are each projected (3×3) and summed into merged; two pre-activation Conv
blocks (BN→ReLU→Conv) refine, with an identity shortcut over merged.</small>
:::

## Candidate blocks (considered, not selected)

The architecture sweep also evaluated the encoder and merge blocks below. **None was selected for
the final SkiNet** — they are documented here for completeness; the selection rationale is in
Model selection (next section). (`LocalRefinementEncoder` and the `he1` / `local_refinement` merge
modes exist in the block registry but were not part of the sweep, so they are omitted here.)

### He2 encoder (`He2Encoder`; He et al., ECCV 2016)

Pre-activation residual encoder — BN→ReLU precede each convolution, and a 1×1 stride-2 projection
shortcut `P` (matching the downsampled spatial size and channel count) is added back.
(`use_residual=False` is rejected for this block.)

```text
h = Conv(BN-Act(x))         # downsamples (stride 2)
y = Conv(BN-Act(h)) + P(x)  # P is a 1×1 projection shortcut
```

:::{figure} _static/block_he2_encoder.svg
:alt: He2 pre-activation residual encoder block data-flow diagram
:width: 100%
<small>He2 encoder — pre-activation BN→ReLU→Conv twice, with a 1×1 stride-2 projection shortcut P(x) added
to the refined output.</small>
:::

### SE encoder (`SEEncoder`; He et al., ECCV 2016 + Hu et al., CVPR 2018)

The same pre-activation skeleton as He2, with a Squeeze-and-Excitation block recalibrating the
convolution path (channel-wise attention) before the projection shortcut is added:

```text
h     = Conv(BN-Act(x))      # downsamples (stride 2)
conv2 = Conv(BN-Act(h))      # refines (stride 1)
y     = SE(conv2) + P(x)     # SE-recalibrated output + 1×1 projection shortcut
```

:::{figure} _static/block_se_encoder.svg
:alt: Squeeze-and-Excitation pre-activation residual encoder block data-flow diagram
:width: 100%
<small>SE encoder — He2 skeleton with a Squeeze-and-Excitation channel-attention block on the conv path
(squeeze ratio `se_reduction=16`), summed with the 1×1 stride-2 projection shortcut P(x).</small>
:::

### Classical merge (`ClassicalMerge`; Ronneberger et al., MICCAI 2015)

The original UNet merge, and the only mode that **concatenates** rather than projects-and-sums:
decoder and skip features are concatenated along the channel dimension, then passed through two
Conv-BN-Act blocks with no residual shortcut.

```text
merged = concat([decoder_out, skip], dim=1)
h      = Conv-BN-Act(merged)
y      = Conv-BN-Act(h)
```

:::{figure} _static/block_classical_merge.svg
:alt: Classical concatenation merge block data-flow diagram
:width: 100%
<small>Classical merge — concatenate decoder and skip along channels, then Conv 3×3 → BN → ReLU twice, no
residual shortcut.</small>
:::

### He2 merge (`He2Merge`; He et al., ECCV 2016)

Projection-and-sum merge followed by two-conv pre-activation refinement with an identity shortcut,
**without** an attention gate. This is exactly the refinement reused inside the selected
attention-gated merge — the only difference there is the attention gate applied to the skip first.

```python
merged = conv_skip(skip) + conv_dec(decoder_out)
conv1  = Conv( ReLU(BN(merged)) )
conv2  = Conv( ReLU(BN(conv1))  )
output = conv2 + merged                             # identity shortcut
```

:::{figure} _static/block_he2merge.svg
:alt: He2 merge block data-flow diagram
:width: 100%
<small>He2 merge — the skip and decoder inputs are projected separately, summed, refined by two
pre-activation Conv blocks (BN→ReLU→Conv), then added back via an identity shortcut over the merged
sum.</small>
:::

## Model selection

**Final architecture: classical encoder + attention-gate merge**, chosen via the two-stage
architecture sweep below.

### Architecture sweep (conducted)

The encoder and merge blocks were selected in two stages.

**E1 — single-seed screen.** A 3×3 grid (classical · SE · He2 encoder × classical · He2 ·
attention-gate merge) was swept at four learning rates (1e-4, 3e-4, 6e-4, 1e-3), 36 single-seed
runs, ranked on tail-mean (last-10-epoch) val Dice. `classical` was the strongest encoder at
every LR (+0.010–0.016 over `se`) and `lr = 3e-4` was the best LR for the leading architectures.
The `he2` and `attention_gate` merges tied within single-seed noise (Δ ≈ 0.0006 tail-mean Dice),
so the screen could not separate them and deferred the tie-break to a paired multi-seed run.

**E2 — 10-seed tie-break (decisive).** `classical + attention_gate` (AG) and `classical + he2`
(HE2) were trained on 10 shared seeds (100–109) at lr = 3e-4, analysing the paired differences
Δ = AG − HE2. On the pre-registered primary metric — plateau Dice (mean of the last 10 epochs) —
**AG wins: 0.8300 vs 0.8275 (Δ +0.0025, Wilcoxon p = 0.037, BCa 95 % CI [+0.0006, +0.0048],
d_z = +0.70, 8/10 seeds)**. Peak Dice and IoU are statistical ties (|Δ| < 0.001, p > 0.7).
HE2's only confirmed advantage is throughput (135.3 vs 119.7 samples/s, +13 %, p = 0.002).
The winning combination is **classical encoder + attention_gate merge**.

### Model-selection figures

The figures below summarise the 10-seed tie-break between the two finalist
configurations (classical/attention-gate vs classical/He2).

### Figure 1 — Paired slope graph

Each line connects one seed's val Dice under both configurations. The slope
direction shows which performed better for that seed.

:::{figure} _static/model_selection/E2_fig1_paired_slopegraph.png
:alt: Paired slope graph comparing classical/attention-gate and classical/He2 val Dice across 10 seeds
:width: 680px
:align: center
<small>Fig 1. Per-seed val Dice for the two finalist configurations across the 10-seed tie-break.
Plateau Dice tilts to attention-gate (8/10 seeds); peak Dice is an even split.</small>
:::

### Figure 2 — Forest plot of paired differences

:::{figure} _static/model_selection/E2_fig2_forest_paired_diff.png
:alt: Forest plot of per-seed plateau-Dice differences (attention-gate − He2)
:width: 680px
:align: center
<small>Fig 2. Forest plot of paired plateau-Dice differences (AG − He2) across 10 seeds; pooled mean
+0.0025 (BCa 95 % CI [+0.0006, +0.0048], excludes zero).</small>
:::

## Training setup

| Setting | Value |
|---------|-------|
| Dataset | ISIC 2017 — 2 000 train / 150 val / 600 test dermoscopic images |
| Input resolution | 256×256 px (the expected input size), resized offline; normalised with dataset-computed mean & std. The network is fully convolutional and accepts any H×W divisible by 16 (the 4 stride-2 downsampling stages), but 256×256 is what the model is trained and deployed at and what the exported ONNX graph fixes. |
| Batch size | 8 per GPU × 2 GPUs = 16 effective (DDP, ddp_spawn) |
| Max epochs | 200 |
| Loss | BCE-Dice: 0.5 × BCEWithLogitsLoss + 0.5 × Dice loss |
| Optimiser | Adam (β₁=0.9, β₂=0.999, ε=1e-8, weight decay=0) |
| Learning rate | 3×10⁻⁴ |
| LR schedule | None in the final training setup|
| Precision | 16-bit mixed (AMP, auto-set from accelerator) |
| Hardware | 2 × NVIDIA T4 (Kaggle), PyTorch Lightning |
| Weight init | Kaiming normal (fan_in, ReLU) for all Conv2d/ConvTranspose2d; BN weights ~ N(1, 0.01) |
| Checkpoint | Best val Dice saved; optimal sigmoid threshold stored in checkpoint buffer |

Key design decisions and their supporting experiments:

| Decision | Experiment | Notebook |
|---|---|---|
| Batch size | E0 batch size sweep | [E0-batch-size-sweep-analysis-unet2d-isic2017.ipynb](E0-batch-size-sweep-analysis-unet2d-isic2017.ipynb) |
| Learning rate | E1 LR sweep (lr in [1e-3, 6e-4, 3e-4, 1e-4]) | [E1-isic2017-unet2d-modelsw-summary-all-lr.ipynb](E1-isic2017-unet2d-modelsw-summary-all-lr.ipynb) |
| Architecture (encoder/merge modes) | E2 10-seed tiebreak | [E2-isic2017-unet2d-model-tiebreak-10seed.ipynb](E2-isic2017-unet2d-model-tiebreak-10seed.ipynb) |
| LR scheduler | E3 cosine annealing vs ReduceOnPlateau | [E3-isic2017-unet2d-lr-decay-study.ipynb](../../analysis_results/E3-isic2017-unet2d-lr-decay-study.ipynb) |
| Threshold | E4 threshold sweep | [E4-isic2017-unet2d-threshold-selection.ipynb](E4-isic2017-unet2d-threshold-selection.ipynb) |
| Production model | EF model selection | [EF_isic2017_unet2d_E4_production_model_selection.ipynb](EF_isic2017_unet2d_E4_production_model_selection.ipynb) |

**Batch size rationale.** Throughput sweep (bs 2→64, 2×T4): bs=8 is the smallest batch on the
plateau (≥81% of peak throughput in both augmented and non-augmented conditions) and the last
point before the time-per-step inflection. Augmentations add negligible cost at this size.
Peak GPU memory: 0.43 GB/GPU.

**Learning rate rationale.** Joint architecture × LR sweep (E1): the 3×3 encoder × merge grid
trained at four LRs, ranked on tail-mean (last-10-epoch) val Dice. lr=3×10⁻⁴ was best for the
`classical` encoder and the modal winner overall (top LR for 6 of 9 architectures). It also gave the
most LR-robust result for the leading architectures, so no LR schedule was needed at this point.

**Architecture rationale.** E1 eliminated all but the `classical` encoder and two merge finalists,
`he2` and `attention_gate`, separated by noise. E2 broke the tie with a paired 10-seed test (shared
init, both at lr=3×10⁻⁴): on the pre-registered plateau-Dice metric `attention_gate` wins (Δ +0.0025,
p = 0.037, 8/10 seeds); peak accuracy is a tie and `he2`'s only edge is ~13 % faster training. CIs
are an optimistic bound, since all seeds reuse one fixed split. **Locked: `classical` encoder +
`attention_gate` merge at lr=3×10⁻⁴.**

**LR schedule rationale.** Scheduler sweep (cosine annealing vs ReduceLROnPlateau, single seed):
cosine annealing 0.8635 val Dice at epoch 142; ReduceLROnPlateau 0.8625 at epoch 102.
Margin 0.001 is below the 0.01 practical-significance threshold. Flat-LR baseline pending re-run.

## Augmentation pipeline (training only)

| Type | Transforms |
|------|-----------|
| Spatial | Square symmetry (D₄ group flips/rotations), affine (scale · translate · rotate ±20°), perspective, elastic deformation |
| Photometric | Colour jitter (brightness, contrast, saturation, hue), Gaussian blur, Gaussian noise |
| Normalisation | Per-channel standardisation: μ = [0.699, 0.556, 0.5121], σ = [0.1576, 0.1562, 0.1706].
Stats are computed on the raw uint8 images (before any augmentation) from the training split only.|

## Inference pipeline

1. Image resized to 256×256 and normalised with the training-set statistics above.
2. Forward pass through U-Net; raw logits passed through sigmoid → probabilities ∈ [0, 1].
3. Threshold of 0.5 is applied to produce the binary mask. E4 experiment [E4-isic2017-unet2d-threshold-selection.ipynb](E4-isic2017-unet2d-threshold-selection.ipynb)
 evaluated whether replacing the default τ = 0.5 with a validation-tuned threshold τ* would have yielded an improvement.
4. Mask returned to the caller at 256×256; the web app overlays it on the original image.
