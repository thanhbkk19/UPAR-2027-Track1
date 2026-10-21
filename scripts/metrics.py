"""UPAR metrics: strict >0.5 decisions and equal weighting of source datasets."""
import numpy as np


def score_dataset(labels, probabilities):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if labels.shape != probabilities.shape or labels.ndim != 2:
        raise ValueError('Expected equal N x C arrays')
    if not np.isfinite(probabilities).all():
        raise ValueError('Nonfinite predictions')
    # Invalid attributes are excluded, as specified on the competition page.
    keep = np.all((labels == 0) | (labels == 1), axis=0)
    labels, pred = labels[:, keep] == 1, probabilities[:, keep] > 0.5
    if not labels.shape[1] or not labels.shape[0]:
        raise ValueError('No valid labels')
    eps = 1e-20
    tp = (labels & pred).sum(0).astype(float)
    tn = (~labels & ~pred).sum(0).astype(float)
    pos, neg = labels.sum(0), (~labels).sum(0)
    # Match the supplied baseline report: average only supported polarities.
    # Also expose the zero-fill variant until private scorer parity is verified.
    tpr, tnr = tp / (pos + eps), tn / (neg + eps)
    ma = ((tpr + tnr) / ((pos > 0).astype(float) + (neg > 0))).mean()
    ma_zero_fill = ((tpr + tnr) * 0.5).mean()
    lf1 = (2 * tp / (pos + pred.sum(0) + eps)).mean()
    intersection = (labels & pred).sum(1)
    precision = (intersection / (pred.sum(1) + eps)).mean()
    recall = (intersection / (labels.sum(1) + eps)).mean()
    f1 = 2 * precision * recall / (precision + recall + eps)
    return dict(mA=float(ma), mA_zero_fill=float(ma_zero_fill), Label_F1=float(lf1),
                Instance_Acc=float((intersection / ((labels | pred).sum(1) + eps)).mean()),
                Instance_Prec=float(precision), Instance_Rec=float(recall),
                Instance_F1=float(f1), Challenge_Avg=float(2 * ma * f1 / (ma + f1 + eps)),
                Challenge_Avg_zero_fill=float(2 * ma_zero_fill * f1 / (ma_zero_fill + f1 + eps)))


def score(labels, probabilities, domains):
    domains = np.asarray(domains)
    per_dataset = {str(d): score_dataset(labels[domains == d], probabilities[domains == d])
                   for d in np.unique(domains)}
    keys = next(iter(per_dataset.values())).keys()
    return dict(per_dataset=per_dataset,
                mean={k: float(np.mean([r[k] for r in per_dataset.values()])) for k in keys},
                std={k: float(np.std([r[k] for r in per_dataset.values()])) for k in keys})


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))
