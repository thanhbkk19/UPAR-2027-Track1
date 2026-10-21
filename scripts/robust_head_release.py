"""Fit and export the fixed v15 committee on the existing v11 WiSE features.

No backbone retraining, new external images, or inference-time VLM is needed.
The VLM head learns only from cached general-foundation-model answers on
organizer training images. Selection and source-excluded audits are separate.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from clip_large import ClipLarge, RegionHead
from robust_head_screen import RECIPES, SmoothRegionHead, fit_head


class RobustCommittee(nn.Module):
    def __init__(self, model, heads, teacher_head, columns, teacher_weight=.2, packed=False, original_weight=0.):
        super().__init__()
        self.model = model
        self.heads = nn.ModuleList(heads)
        self.teacher_head = teacher_head
        self.register_buffer('columns', columns)
        self.teacher_weight = teacher_weight
        self.packed = packed
        self.original_weight = original_weight
        if packed:
            # A regional linear readout is a sparse linear map of the five
            # pooled tokens. Stack all four classifiers into one GEMM.
            # Dropout is disabled at inference, so this changes no learned rule.
            dim = model.head.dim
            bank = nn.Linear(5*dim, (len(heads)+1)*40)
            bank.weight.data.zero_()
            for k, head in enumerate(list(heads)+[teacher_head]):
                base = head.base if isinstance(head, SmoothRegionHead) else head
                bank.weight.data[k*40:(k+1)*40, :dim].copy_(base.global_head.weight)
                bank.bias.data[k*40:(k+1)*40].copy_(base.global_head.bias)
                for j, region in enumerate(base.region_index.tolist()):
                    start = (1+region)*dim
                    bank.weight.data[k*40+j, start:start+dim].copy_(base.local_weight[j])
            self.bank = bank
            self.head_count = len(heads)

    def forward(self, pixels):
        x = self.model.features(pixels)
        if self.packed:
            logits = self.bank(x.flatten(1)*self.model.head.dim**.5).reshape(x.shape[0], self.head_count+1, 40)
            p = logits[:, :self.head_count].float().sigmoid().mean(1).clamp(1e-7, 1-1e-7)
            teacher_logits = logits[:, self.head_count]
        else:
            p = torch.stack([h(x).float().sigmoid() for h in self.heads]).mean(0).clamp(1e-7, 1-1e-7)
            teacher_logits = self.teacher_head(x)
        z = torch.log(p)-torch.log1p(-p)
        pt = teacher_logits.float().sigmoid().clamp(1e-7, 1-1e-7)
        zt = torch.log(pt)-torch.log1p(-pt)
        delta = (zt.index_select(1, self.columns)-z.index_select(1, self.columns))*self.teacher_weight
        z = z + torch.zeros_like(z).index_copy(1, self.columns, delta)
        if self.original_weight:
            original = self.model.head(x).float().sigmoid()
            p = ((1-self.original_weight)*z.sigmoid()+self.original_weight*original).clamp(1e-7, 1-1e-7)
            z = torch.log(p)-torch.log1p(-p)
        return z


def fit(args):
    cache = Path(args.cache)
    labels = np.load(cache/'train_labels.npz')
    paths = labels['paths']
    domains = np.array([p.split('/')[0] for p in paths])
    x = torch.from_numpy(np.load(cache/'train_features.npy')).cuda()
    y = torch.from_numpy(labels['labels']).cuda()
    heads = []
    for source in sorted(set(domains)):
        ii = torch.from_numpy(np.flatnonzero(domains == source)).cuda()
        h = fit_head(x[ii], y[ii], RECIPES['mixup'])
        heads.append({k: v.detach().cpu() for k, v in h.state_dict().items()})
        print(json.dumps(dict(source=source, status='fit')), flush=True)
    with np.load('artifacts/loso/vlm_train24k.npz') as file:
        vlm = {k: file[k] for k in file.files}
    positions = {p: i for i, p in enumerate(paths)}
    keep = [i for i, p in enumerate(vlm['paths']) if p in positions]
    xv = x[[positions[vlm['paths'][i]] for i in keep]].float()
    yv = torch.from_numpy(1/(1+np.exp(-np.clip(vlm['scores'][keep], -60, 60)))).cuda().float()
    columns = torch.tensor([list(labels['names']).index(a) for a in vlm['attributes']], device='cuda')
    torch.manual_seed(0)
    teacher = RegionHead(True, x.shape[-1]).cuda()
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=3e-3, weight_decay=.01)
    for _ in range(12):
        teacher.train()
        order = torch.randperm(len(xv), device='cuda')
        for start in range(0, len(xv), 1024):
            ii = order[start:start+1024]
            loss = F.binary_cross_entropy_with_logits(teacher(xv[ii])[:, columns], yv[ii])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state['robust_committee'] = dict(heads=heads, sources=sorted(set(domains)),
        teacher_head={k: v.detach().cpu() for k, v in teacher.state_dict().items()},
        teacher_columns=columns.cpu(), teacher_weight=.2, teacher_training_paths=vlm['paths'][keep],
        recipe='Per-source RegionHead, mixup alpha 0.4, seed 0, 12 epochs, AdamW lr 3e-3; 20% VLM head in logit space on 16 attributes',
        teacher='Qwen3-VL-8B general foundation model; cached yes/no scores on organizer images only',
        cache=str(cache), cache_manifest=json.loads((cache/'manifest.json').read_text()))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)
    print(json.dumps(dict(output=str(output), teacher_images=len(keep))), flush=True)


def export(args):
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = state['config']
    a = cfg['args']
    size = (a.get('height', 224), a.get('width', 224))
    model = ClipLarge(a['tag'], regional=cfg['regional'], size=size, name=a.get('timm_name'))
    model.load_state_dict(state['ema'], strict=True)
    c = state['robust_committee']
    inference_size = c.get('cache_manifest', {}).get('args', {}).get('inference_size')
    if inference_size:
        from resolution_probe import resize_model
        size = tuple(inference_size)
        resize_model(model, size)
    heads = []
    for hs in c['heads']:
        h = SmoothRegionHead(model.head.dim, 0)
        h.load_state_dict(hs)
        heads.append(h)
    teacher = RegionHead(True, model.head.dim)
    teacher.load_state_dict(c['teacher_head'])
    if not 0 <= args.original_weight <= 1:
        raise ValueError('Original classifier weight must be between zero and one')
    net = RobustCommittee(model, heads, teacher, c['teacher_columns'], c['teacher_weight'], packed=args.packed,
                          original_weight=args.original_weight).eval().cuda()
    reference = copy.deepcopy(net).eval()
    net.model.backbone.half()
    net.model.mean = net.model.mean.half()
    net.model.std = net.model.std.half()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        traced = torch.jit.trace(net, torch.zeros(2, 3, *size, device='cuda', dtype=torch.float16), check_trace=False)
        traced.save(str(output))
        loaded = torch.jit.load(str(output), map_location='cuda').eval()
        pixels = torch.rand(17, 3, *size, device='cuda')*2-1
        with torch.autocast('cuda', dtype=torch.float16):
            expected = reference(pixels).float().sigmoid()
        actual = loaded(pixels.half()).float().sigmoid()
        difference = float((expected-actual).abs().max())
        if not torch.isfinite(actual).all() or difference > .01:
            raise ValueError(f'Export probability parity failed: {difference}')
    report = dict(checkpoint=args.checkpoint, output=str(output), sources=c['sources'],
                  max_probability_difference=difference, recipe=c['recipe'],
                  teacher_images=len(c['teacher_training_paths']), inference_vlm=False, packed_heads=args.packed,
                  image_size=list(size), original_weight=args.original_weight)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['fit', 'export'])
    parser.add_argument('--cache')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--packed', action='store_true')
    parser.add_argument('--original-weight', type=float, default=0.)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20261014)
    (fit if args.action == 'fit' else export)(args)
