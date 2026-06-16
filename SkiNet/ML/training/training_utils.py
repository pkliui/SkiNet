import torch


def mean_dice_per_image(probs: torch.Tensor, targets: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Compute per-image Dice (= binary F1) and return the mean across all images.

    Vectorised: no Python loop over images.

    :param probs: float tensor of shape [N, pixels] — sigmoid probabilities per image
    :param targets: long or bool tensor of shape [N, pixels] — ground-truth binary masks
    :param threshold: scalar threshold applied to probs to obtain binary predictions
    :return: scalar tensor — mean Dice averaged over N images
    """
    preds = (probs >= threshold)
    tgt = targets.bool()
    tp = (preds & tgt).sum(dim=1).float()
    fp = (preds & ~tgt).sum(dim=1).float()
    fn = (~preds & tgt).sum(dim=1).float()
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return dice.mean()


def find_best_threshold(probs: torch.Tensor,
                        targets: torch.Tensor,
                        n_thresholds: int = 51) -> dict[str, float]:
    """
    Find the optimal threshold that maximizes the Dice (F1) score for binary predictions.

    This function evaluates multiple thresholds in a fully vectorized manner on the GPU,
    avoiding Python loops and repeated device synchronization.

    :param probs: 1D tensor of predicted probabilities with shape [N]. Must be on the same device
        where computation should occur (CPU or GPU).
    :param targets: 1D tensor of ground-truth binary labels with shape [N]. Values will be converted to boolean internally.
    :param n_thresholds: Number of evenly spaced thresholds in the range [0.0, 1.0]. Defaults to 51.
    :return: A dict with keys ``"best_threshold"`` (float) and ``"best_dice"`` (float).
    """
    # e.g. torch.Size([51])
    thresholds = torch.linspace(1.0, 0.0, n_thresholds, device=probs.device)

    # targets and probs.shape [N], thresholds.shape e.g. [51]
    # probs.unsqueeze(0): [1, N], thresholds.unsqueeze(1): [51, 1]
    # after unsqueeze: [1, N] >= [51, 1]
    # after broadcast preds are: [51, N] bool mask
    preds = probs.unsqueeze(0) >= thresholds.unsqueeze(1)
    targets = targets.bool().unsqueeze(0)  # [1, N]

    # [51, N] & [1, N] ->[51, N] & [51, N] -> [51, N]
    # sum(dim=1):  [51, N] -> [51]
    tp = (preds & targets).sum(dim=1).float()
    fp = (preds & ~targets).sum(dim=1).float()
    fn = (~preds & targets).sum(dim=1).float()
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)  # [51]

    # Thresholds are ordered high→low, so argmax returns the first (highest) threshold
    # in case multiple thresholds achieve the same maximum Dice coefficient.
    best_idx = dice.argmax()

    # Only two item calls, minimizing GPU-CPU synchronization overhead
    return {"best_threshold": thresholds[best_idx].item(), "best_dice": dice[best_idx].item()}
