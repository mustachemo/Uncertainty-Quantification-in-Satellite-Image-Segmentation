import torch
import torch.nn as nn
import torch.nn.functional as F
from .dice_score import dice_loss, multiclass_dice_coeff


def dice_coefficient(y_pred, y_true, smooth=1e-6):
    """Dice coefficient for binary segmentation"""
    y_pred = torch.sigmoid(y_pred)
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)

    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f) + torch.sum(y_pred_f)
    return (2.0 * intersection + smooth) / (union + smooth)


def combined_loss(y_pred, y_true):
    """Combined loss function that combines Dice loss with binary cross-entropy"""
    # y_pred is expected to be logits
    bce = F.binary_cross_entropy_with_logits(y_pred, y_true)
    # convert y_pred to probabilities for dice loss
    y_pred_probs = torch.sigmoid(y_pred)
    d_loss = 1 - multiclass_dice_coeff(y_pred_probs, y_true.long())
    return bce + d_loss


def uncertainty_aware_loss(y_pred, y_true, lambda_=0.01):
    """Uncertainty-aware loss function that combines Dice loss with uncertainty"""
    # y_pred is expected to be logits
    bce = F.binary_cross_entropy_with_logits(y_pred, y_true)
    y_pred_probs = torch.sigmoid(y_pred)
    d_loss = 1 - multiclass_dice_coeff(y_pred_probs, y_true.long())

    # The original implementation calculates std over the whole prediction tensor.
    # This is a measure of prediction variance, not epistemic uncertainty.
    # For MC dropout, one would typically calculate variance over multiple forward passes.
    # Replicating original behaviour:
    uncertainty = torch.std(y_pred_probs)

    base_loss = bce + d_loss
    weighted_loss = base_loss * (1 + lambda_ * uncertainty)

    return weighted_loss.mean()


#################### Bayesian U-Net Loss Functions ####################
def bayesian_unet_nll(y_pred_dist, y_true):
    """
    Negative log-likelihood for Bayesian U-Net.
    `y_pred_dist` is a torch.distributions.Distribution object (e.g., Bernoulli).
    """
    return -y_pred_dist.log_prob(y_true).mean()


def combined_loss_bayesian_unet(y_pred_dist, y_true):
    """Combine the Dice loss with negative log-likelihood for probabilistic layers."""
    nll_loss = bayesian_unet_nll(y_pred_dist, y_true)
    # y_pred_dist.mean is the mean of the distribution, which for Bernoulli is the probability.
    dice_loss_value = 1 - multiclass_dice_coeff(y_pred_dist.mean, y_true.long())
    return dice_loss_value + nll_loss