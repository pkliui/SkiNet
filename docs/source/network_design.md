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
Skip connections (not shown) run from each encoder layer to its mirror decoder merge block.
Each upsampling step uses a transposed convolution (stride 2), restoring the spatial dimensions
halved by the corresponding encoder stage.
:::

## Block design

### Encoder block — Classical (Ronneberger et al., 2015)

Two sequential Conv2d(3×3) → BatchNorm → ReLU layers per encoder stage.
The first convolution downsamples (stride 2); the second refines at the
reduced resolution (stride 1). No residual connection — the original
UNet design, used as the baseline in the architecture sweep.

:::{figure} _static/block_encoder.svg
:alt: Classical encoder block data-flow diagram
:width: 100%
Classical encoder block — Conv 3×3 (stride 2) → BN → ReLU → Conv 3×3 (stride 1) → BN → ReLU.
:::

### Decoder block — transposed convolution upsampling

Each decoder stage begins by doubling the spatial resolution with a single
transposed convolution, followed by BatchNorm and ReLU. The kernel size is
set to `encoder_kernel × encoder_stride = 3 × 2 = 6` rather than
the more common 2×2, which eliminates the uneven overlap pattern that
produces checkerboard artefacts in the output mask (Odena et al., 2016).
The stride of 2 exactly inverts the downsampling applied by the corresponding
encoder stage. The output of this block is passed directly to the He2 Merge
block below alongside the skip connection.

:::{figure} _static/block_decoder.svg
:alt: Decoder upsampling block data-flow diagram
:width: 100%
Decoder upsampling block — ConvTranspose2d 6×6 (stride 2, spatial ×2, channels ÷2) → BN → ReLU.
:::

### Merge block — He2 pre-activation (He et al., ECCV 2016)

The skip-connection tensor and the upsampled decoder tensor are each
projected through separate 3×3 convolutions, then _summed_
(rather than concatenated). This is algebraically equivalent to
concatenation+convolution but avoids materialising the doubled-channel
tensor in memory. Two BN→ReLU→Conv refinement convolutions follow,
with an identity shortcut over the merged sum:

```python
merged  = conv_skip(skip) + conv_dec(decoder_out)
conv1   = Conv( ReLU(BN(merged)) )
conv2   = Conv( ReLU(BN(conv1))  )
output  = conv2 + merged          # identity shortcut
```

:::{figure} _static/block_he2merge.svg
:alt: He2 merge block data-flow diagram
:width: 100%
He2 merge block — skip and decoder inputs projected separately, summed, refined by two pre-activation
Conv blocks, then added back via an identity shortcut.
:::

### Architecture sweep (conducted)

A 3×3 grid (classical · He2 · SE encoder × classical · He2 · attention-gate merge)
was swept over 200 epochs, 2×T4 DDP, Adam lr=3×10⁻⁴, full augmentation.
Rankings use the tail-mean of val Dice over the last 50 epochs:
classical/He2 0.8409 (rank 1), classical/attention-gate 0.8351 (rank 2), SE/classical
~0.835 (rank 3). The 0.006-Dice margin triggered a 5-seed tie-break; the
paired t-test (p=0.18, mean diff +0.0016) was non-significant, so the full-grid rank-1
result was adopted by the conservative selection rule.
The encoder choice drove more variance (marginal spread 0.006) than the merge block (0.003).
The winning combination is **classical encoder + He2 merge**.

## Model Selection

The figures below summarise the 10-seed tie-break between the two top-ranked
configurations (classical/He2 vs classical/attention-gate).

### Figure 1 — Paired slope graph

Each line connects one seed's val Dice under both configurations. The slope
direction shows which performed better for that seed.

:::{figure} _static/model_selection/E2_fig1_paired_slopegraph.png
:alt: Paired slope graph comparing classical/He2 and classical/attention-gate val Dice across 10 seeds
:width: 680px
:align: center
Fig 1. Per-seed val Dice for the two top-ranked configurations across the 10-seed tie-break.
:::

### Figure 2 — Forest plot of paired differences

:::{figure} _static/model_selection/E2_fig2_forest_paired_diff.png
:alt: Forest plot of per-seed Dice differences between classical/He2 and classical/attention-gate
:width: 680px
:align: center
Fig 2. Forest plot of paired Dice differences (He2 − attention-gate) across 10 seeds.
:::

## Training setup

| Setting | Value |
|---------|-------|
| Dataset | ISIC 2017 — 2 000 train / 150 val / 600 test dermoscopic images |
| Input resolution | 256×256 px, resized offline; normalised with dataset-computed mean & std |
| Batch size | 8 per GPU × 2 GPUs = 16 effective (DDP, ddp_spawn) |
| Max epochs | 300 with early stopping (patience 50, Δ min 0.002 on val Dice) |
| Loss | BCE-Dice: 0.5 × BCEWithLogitsLoss + 0.5 × Dice loss |
| Optimiser | Adam (β₁=0.9, β₂=0.999, ε=1e-8, weight decay=0) |
| Learning rate | 3×10⁻⁴ |
| LR schedule | Cosine annealing (T_max=300, η_min=1×10⁻⁶) |
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

**Learning rate rationale.** 5-point log-spaced sweep [1e-4 … 3e-3], AdamW, 2×T4, 100 epochs.
lr=3×10⁻⁴ achieved the highest mean val Dice (0.806), lowest epoch variance (σ=0.018), and
cleanest convergence. Consistent with two prior Adam sweeps. lr=3×10⁻³ caused clear degradation
(mean Dice −0.018, convergence to 0.80 delayed by 35 epochs).

**LR schedule rationale.** Scheduler sweep (cosine annealing vs ReduceLROnPlateau, single seed):
cosine annealing 0.8635 val Dice at epoch 142; ReduceLROnPlateau 0.8625 at epoch 102.
Margin 0.001 is below the 0.01 practical-significance threshold. Flat-LR baseline pending re-run.

## Augmentation pipeline (training only)

| Type | Transforms |
|------|-----------|
| Spatial | Square symmetry (D₄ group flips/rotations), affine (scale · translate · rotate ±20°), perspective, elastic deformation |
| Photometric | Colour jitter (brightness, contrast, saturation, hue), Gaussian blur, Gaussian noise |
| Normalisation | Per-channel standardisation: μ = [0.699, 0.556, 0.512], σ = [0.158, 0.156, 0.171] |

## Inference pipeline

1. Image resized to 256×256 and normalised with the training-set statistics above.
2. Forward pass through U-Net; raw logits passed through sigmoid → probabilities ∈ [0, 1].
3. Threshold applied to produce the binary mask. The threshold is _not_ fixed
   at 0.5 — at each validation epoch a vectorised sweep over 51 candidate thresholds
   (linspace 0→1) finds the value that maximises Dice on the full validation set.
   The best threshold is stored in the checkpoint and used at test / inference time.
4. Mask returned to the caller at 256×256; the web app overlays it on the original image.
