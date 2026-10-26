"""Label-frequency weighting for the binary cross-entropy."""
import torch


def polarity_weights(rate):
    """Per-attribute weights for positive and negative labels; the rarer polarity gets more weight."""
    return torch.exp(1-rate), torch.exp(rate)
