# Network Design — SkiNet

## Architecture — U-Net 2D (custom, from scratch)

SkiNet is a symmetric encoder–decoder network built entirely from scratch in PyTorch, without pre-trained backbones. The encoder path progressively compresses spatial resolution while doubling channel depth at each stage. The decoder path mirrors it, restoring resolution via transposed convolutions. Skip connections at every resolution level carry high-frequency spatial detail from encoder to decoder, preventing boundary information from being lost during downsampling.

---

## Architecture diagram

```
Encoder 1        Encoder 2        Encoder 3        Encoder 4
16 ch · 256²  →  32 ch · 128²  →  64 ch · 64²  →  128 ch · 32²
stride 1         stride 2         stride 2         stride 2
                                                        ↓
                              Bottleneck (Encoder 5)
                              256 ch · 16×16 · stride 2
                                                        ↓
AG-Merge 4      AG-Merge 3      AG-Merge 2      AG-Merge 1
128 ch · 32²  ←  64 ch · 64²  ←  32 ch · 128²  ←  16 ch · 256²
                                                        ↓
                         1×1 conv → sigmoid
                     1 ch · 256×256 · binary mask
```

Skip connections run from each encoder layer to its mirror decoder merge block. Each upsampling step uses a transposed convolution (stride 2), restoring the spatial dimensions halved by the corresponding encoder stage.

---

## Block design

### Encoder block — Classical (Ronneberger et al., 2015)

Two sequential Conv2d(3×3) → BatchNorm → ReLU layers per encoder stage. The first convolution downsamples (stride 2); the second refines at the reduced resolution (stride 1). No residual connection — the original UNet design, used as the baseline in the architecture sweep.

### Merge block — Attention Gate (AG) (REFERENCE)

DESCRIPTION

### Batch size decision
Notebook to be reviewed
repos/SkiNet/analysis_results/E0_batch_size_sweep_analysis_unet2d_isic2017.ipynb

### Learning rate decision
Notebook to be reviewed
DESCRIPTION as per
repos/SkiNet/analysis_results/E1_isic2017_unet2d_modelsw_summary_all_lr.ipynb

### Architecture sweep

Notebook has been reviewed
repos/SkiNet/analysis_results/E2-isic2017-unet2d-model-tiebreak-10seed.ipynb

### LR decay
repos/SkiNet/analysis_results/E3-isic2017-unet2d-cosanneal-1seed.ipynb
repos/SkiNet/analysis_results/E3-isic2017-unet2d-reduceonplateauON15-1seed1.ipynb

### Threshold Dice score

repos/SkiNet/analysis_results/E4_isic2017_unet2d_threshold_sweep_analysis.ipynb


## Training setup

repos/SkiNet/analysis_results/EF_isic2017_unet2d_E4_production_model_selection.ipynb
