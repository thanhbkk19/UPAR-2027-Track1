"""Training-only label-frequency weighting, shared by frozen and tuned heads."""
import torch


def bce_polarity_weights(rate, balance_power=1., inverse_balance_power=0.):
    if not 0<=inverse_balance_power<=1:raise ValueError('Inverse power must lie in [0,1]')
    positive=torch.exp(balance_power*(1-rate))
    negative=torch.exp(balance_power*rate)
    if inverse_balance_power:
        # A floor prevents singletons from dominating; expected per-label loss
        # weight is normalized to one, so common/rare labels keep equal scale.
        stable=rate.clamp(.002,.998)
        positive=positive*(.5/stable).pow(inverse_balance_power)
        negative=negative*(.5/(1-stable)).pow(inverse_balance_power)
        normalizer=positive*rate+negative*(1-rate)
        positive=positive/normalizer;negative=negative/normalizer
    return positive,negative


def soft_domain_f1_loss(logits,targets,ids,domain_count):
    probabilities=logits.float().sigmoid();intersection=(probabilities*targets).sum(1)
    precision=intersection/probabilities.sum(1).clamp_min(1)
    recall=intersection/targets.sum(1).clamp_min(1)
    n=torch.bincount(ids,minlength=domain_count).float()
    p=torch.zeros_like(n).scatter_add_(0,ids,precision)/n.clamp_min(1)
    r=torch.zeros_like(n).scatter_add_(0,ids,recall)/n.clamp_min(1)
    f1=2*p*r/(p+r).clamp_min(1e-6);present=(n>0).float()
    return ((1-f1)*present).sum()/present.sum().clamp_min(1)
