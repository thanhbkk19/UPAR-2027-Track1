"""Compare fixed source-head recipes on source-excluded backbone features.

The held source is excluded from every head fit. Reference paths are explicitly
aligned. Save heads and probabilities so a later, separate audit can reproduce
the comparison without training again.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from attribute_losses import bce_polarity_weights
from clip_large import RegionHead
from fold_eval import FOLDS, fold_prior, logit
from metrics import score_dataset, sigmoid


class SmoothRegionHead(nn.Module):
    def __init__(self, dim=1024, hidden=0):
        super().__init__()
        self.base = RegionHead(True, dim)
        self.hidden = hidden
        if hidden:
            self.global_mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                            nn.Dropout(.25), nn.Linear(hidden, 40))
            self.local_mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                           nn.Dropout(.25), nn.Linear(hidden, 40))
            for block in (self.global_mlp, self.local_mlp):
                nn.init.zeros_(block[-1].weight)
                nn.init.zeros_(block[-1].bias)

    def forward(self, x):
        z = self.base(x)
        if self.hidden:
            h = x * self.base.dim ** .5
            global_z = self.global_mlp(h[:, 0])
            local_z = self.local_mlp(h[:, 1:])
            routed = local_z.gather(1, self.base.region_index.view(1, 1, -1).expand(x.shape[0], 1, -1))[:, 0]
            z = z + global_z + routed
        return z


RECIPES = {
    'linear': dict(hidden=0, mixup=0.),
    'mixup': dict(hidden=0, mixup=.4),
    'mlp': dict(hidden=256, mixup=0.),
    'mlp_mixup': dict(hidden=256, mixup=.4),
}


def fit_head(x, y, recipe):
    torch.manual_seed(0)
    model = SmoothRegionHead(x.shape[-1], recipe['hidden']).cuda()
    pw, nw = bce_polarity_weights(y.mean(0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=.01)
    rng = np.random.default_rng(0)
    for _ in range(12):
        model.train()
        order = torch.randperm(len(x), device=x.device)
        for start in range(0, len(x), 1024):
            ii = order[start:start+1024]
            xx, yy = x[ii].float(), y[ii]
            if recipe['mixup']:
                lam = float(rng.beta(recipe['mixup'], recipe['mixup']))
                jj = torch.randperm(len(ii), device=x.device)
                xx = xx * lam + xx[jj] * (1-lam)
                yy = yy * lam + yy[jj] * (1-lam)
            z = model(xx)
            loss = (F.binary_cross_entropy_with_logits(z, yy, reduction='none') *
                    (yy * pw + (1-yy) * nw)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval()


def predict(model, x):
    with torch.inference_mode():
        return torch.cat([model(x[i:i+4096].float()).sigmoid()
                          for i in range(0, len(x), 4096)]).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='artifacts/robust_heads_v15')
    parser.add_argument('--recipes', nargs='+', default=list(RECIPES))
    parser.add_argument('--cache-template', default='artifacts/features/foldwise_{member}_{held}')
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    table = {r: {} for r in args.recipes}
    for held in FOLDS:
        ref = np.load(f'artifacts/loso/wise2_pe_{held}.npz')
        probs = {r: [] for r in args.recipes}
        for member in ('pe', 'eva'):
            cache = Path(args.cache_template.format(member=member, held=held))
            train = np.load(cache/'train_labels.npz')
            tune = np.load(cache/'tune_labels.npz')
            paths = list(train['paths'])
            domains = np.array([p.split('/')[0] for p in paths])
            x = torch.from_numpy(np.load(cache/'train_features.npy')).cuda()
            y = torch.from_numpy(train['labels']).cuda()
            train_pos = {p: i for i, p in enumerate(paths)}
            tune_pos = {p: i for i, p in enumerate(tune['paths'])}
            if all(p in tune_pos for p in ref['paths']):
                xt = torch.from_numpy(np.load(cache/'tune_features.npy')[[tune_pos[p] for p in ref['paths']]]).cuda()
            else:
                assert all(p in train_pos for p in ref['paths'])
                xt = x[[train_pos[p] for p in ref['paths']]]
            sources = sorted(set(domains)-{held})
            for name in args.recipes:
                predictions, states = [], []
                for source in sources:
                    ii = torch.from_numpy(np.flatnonzero(domains == source)).cuda()
                    model = fit_head(x[ii], y[ii], RECIPES[name])
                    predictions.append(predict(model, xt))
                    states.append({k: v.detach().cpu() for k, v in model.state_dict().items()})
                p = np.mean(predictions, axis=0)
                probs[name].append(p)
                torch.save(dict(heads=states, sources=sources, recipe=RECIPES[name],
                                cache=str(cache), excluded_source=held), out/f'{name}_{member}_{held}.pt')
                np.savez_compressed(out/f'{name}_{member}_{held}.npz',
                                    paths=ref['paths'], labels=ref['labels'], probabilities=p)
                print(json.dumps(dict(fold=held, member=member, recipe=name, status='fit')), flush=True)
            del x, y, xt
        for name in args.recipes:
            z = logit(np.clip(np.mean(probs[name], axis=0), 1e-7, 1-1e-7))
            metrics = score_dataset(ref['labels'], sigmoid(z-.4*logit(fold_prior(held))+.25))
            table[name][held] = metrics
        print(json.dumps(dict(fold=held, scores={n: table[n][held]['Challenge_Avg'] for n in table})), flush=True)
        (out/'screen.json').write_text(json.dumps(table, indent=2))
    print(json.dumps({n: float(np.mean([v['Challenge_Avg'] for v in folds.values()]))
                      for n, folds in table.items()}), flush=True)


if __name__ == '__main__':
    main()
