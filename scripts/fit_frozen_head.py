"""Fit the regional head on frozen pretrained features (initialises fine-tuning and WiSE)."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from losses import polarity_weights
from metrics import score, sigmoid
from model import RegionHead


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--domain-power', type=float, default=.5, help='Row weight = source size ** -power')
    p.add_argument('--exclude-source', help='LOSO: drop this source from the cached training rows')
    p.add_argument('--select', choices=['best', 'last'], default='best', help='best = highest tune Challenge_Avg')
    p.add_argument('--clip-grad', type=float, default=5.)
    p.add_argument('--seed', type=int, default=2092)
    args = p.parse_args()
    torch.set_num_threads(4)
    cache, out = Path(args.cache), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    train = np.load(cache/'train_labels.npz')
    train_domains = np.array([q.split('/')[0] for q in train['paths']])
    rows = np.flatnonzero(train_domains != args.exclude_source)
    x = torch.from_numpy(np.load(cache/'train_features.npy')[rows]).cuda()
    y = torch.from_numpy(train['labels'][rows]).cuda()
    train_domains = train_domains[rows]
    counts = {d: int((train_domains == d).sum()) for d in np.unique(train_domains)}
    dw = np.array([counts[d]**-args.domain_power for d in train_domains], dtype=np.float32)
    domain_weights = torch.from_numpy(dw/dw.mean()).cuda()
    rate = (y*domain_weights[:, None]).sum(0)/domain_weights.sum()
    positive_weight, negative_weight = polarity_weights(rate)
    if args.select == 'best':
        tune = np.load(cache/'tune_labels.npz')
        vx = torch.from_numpy(np.load(cache/'tune_features.npy')).cuda()
        tune_domains = np.array([q.split('/')[0] for q in tune['paths']])

    torch.manual_seed(args.seed)
    head = RegionHead(x.shape[-1]).cuda()
    with torch.no_grad():
        head.global_head.bias.copy_(torch.logit(rate.clamp(.001, .999)))
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=.01)
    best = -1
    for epoch in range(args.epochs):
        head.train()
        order = torch.randperm(len(x), device='cuda')
        for group in optimizer.param_groups:
            group['lr'] = args.lr*(.1+.9*.5*(1+math.cos(math.pi*epoch/args.epochs)))
        for start in range(0, len(x), args.batch_size):
            ii = order[start:start+args.batch_size]
            target = y[ii]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = head(x[ii])
                weight = target*positive_weight+(1-target)*negative_weight
                loss = ((F.binary_cross_entropy_with_logits(logits.float(), target, reduction='none')*weight).mean(1)
                        * domain_weights[ii]).mean()
            loss.backward()
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(head.parameters(), args.clip_grad)
            optimizer.step()
        value = None
        if args.select == 'best':
            head.eval()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                pred = torch.cat([head(vx[s:s+1024]).float() for s in range(0, len(vx), 1024)]).cpu().numpy()
            value = score(tune['labels'], sigmoid(pred), tune_domains)['mean']['Challenge_Avg']
        print(json.dumps(dict(epoch=epoch+1, tune_score=value)), flush=True)
        if args.select == 'last' or value > best:
            best = value if value is not None else best
            torch.save(dict(head=head.state_dict(), epoch=epoch+1, args=vars(args)), out/'head.pt')
    (out/'summary.json').write_text(json.dumps(dict(args=vars(args), best_tune_score=best), indent=2))


if __name__ == '__main__':
    main()
