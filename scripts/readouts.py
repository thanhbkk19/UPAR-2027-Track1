"""Per-source committee + teacher readout on frozen WiSE features, and TorchScript export.

fit:    one SourceHead per training source (feature mixup) and one teacher head on the
        cached VLM scores; stored inside a copy of the WiSE checkpoint.
export: trace the Committee (FP16 backbone) and check it against the eager model.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from losses import polarity_weights
from model import Committee, RegionHead, SourceHead, load_encoder


def fit_source_head(x, y, mixup=.4, epochs=12, lr=3e-3, batch=1024):
    torch.manual_seed(0)
    head = SourceHead(x.shape[-1]).cuda()
    pw, nw = polarity_weights(y.mean(0))
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=.01)
    rng = np.random.default_rng(0)
    for _ in range(epochs):
        head.train()
        order = torch.randperm(len(x), device=x.device)
        for start in range(0, len(x), batch):
            ii = order[start:start+batch]
            xx, yy = x[ii].float(), y[ii]
            lam = float(rng.beta(mixup, mixup))
            jj = torch.randperm(len(ii), device=x.device)
            xx, yy = xx*lam+xx[jj]*(1-lam), yy*lam+yy[jj]*(1-lam)
            loss = (F.binary_cross_entropy_with_logits(head(xx), yy, reduction='none')*(yy*pw+(1-yy)*nw)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return head.eval()


def fit_teacher_head(x, targets, columns, epochs=12, lr=3e-3, batch=1024):
    torch.manual_seed(0)
    head = RegionHead(x.shape[-1]).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=.01)
    for _ in range(epochs):
        head.train()
        order = torch.randperm(len(x), device='cuda')
        for start in range(0, len(x), batch):
            ii = order[start:start+batch]
            loss = F.binary_cross_entropy_with_logits(head(x[ii])[:, columns], targets[ii])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return head.eval()


def cpu_state(module):
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def fit(args):
    cache = Path(args.cache)
    labels = np.load(cache/'train_labels.npz')
    paths = labels['paths']
    domains = np.array([p.split('/')[0] for p in paths])
    x = torch.from_numpy(np.load(cache/'train_features.npy')).cuda()
    y = torch.from_numpy(labels['labels']).cuda()
    sources = sorted(set(domains))
    heads = []
    for source in sources:
        ii = torch.from_numpy(np.flatnonzero(domains == source)).cuda()
        heads.append(cpu_state(fit_source_head(x[ii], y[ii])))
        print(json.dumps(dict(source=source, images=len(ii))), flush=True)
    # Teacher rows outside this cache's training sources (LOSO held-out source) are skipped.
    with np.load(args.teacher) as file:
        teacher = {k: file[k] for k in file.files}
    positions = {p: i for i, p in enumerate(paths)}
    keep = [i for i, p in enumerate(teacher['paths']) if p in positions]
    xt = x[[positions[teacher['paths'][i]] for i in keep]].float()
    targets = torch.from_numpy(1/(1+np.exp(-np.clip(teacher['scores'][keep], -60, 60)))).cuda().float()
    columns = torch.tensor([list(labels['names']).index(a) for a in teacher['attributes']], device='cuda')
    teacher_head = fit_teacher_head(xt, targets, columns)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state['robust_committee'] = dict(
        heads=heads, sources=sources, teacher_head=cpu_state(teacher_head), teacher_columns=columns.cpu(),
        teacher_weight=.2, teacher_training_paths=teacher['paths'][keep],
        recipe='Per-source RegionHead, mixup alpha 0.4, seed 0, 12 epochs, AdamW lr 3e-3; 20% VLM head in logit space on 16 attributes',
        cache=str(cache), cache_manifest=json.loads((cache/'manifest.json').read_text()))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    print(json.dumps(dict(output=args.output, sources=sources, teacher_images=len(keep))), flush=True)


def export(args):
    model, state = load_encoder(args.checkpoint)
    c = state['robust_committee']
    size = (state['config']['args']['height'], state['config']['args']['width'])
    inference_size = c['cache_manifest']['args'].get('inference_size')
    if inference_size:
        size = tuple(inference_size)
        model.set_resolution(size)
    heads = []
    for hs in c['heads']:
        heads.append(SourceHead(model.head.dim))
        heads[-1].load_state_dict(hs)
    teacher = RegionHead(model.head.dim)
    teacher.load_state_dict(c['teacher_head'])
    net = Committee(model, heads, teacher, c['teacher_columns'], c['teacher_weight'], args.original_weight).eval().cuda()
    reference = copy.deepcopy(net).eval()
    net.model.backbone.half()
    net.model.mean, net.model.std = net.model.mean.half(), net.model.std.half()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Profiling JIT fusion is disabled here and in the runtime so FP16 outputs do not change after warm-up.
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
            raise ValueError(f'Export parity failed: {difference}')
    report = dict(checkpoint=args.checkpoint, output=str(output), sources=c['sources'], max_probability_difference=difference,
                  recipe=c['recipe'], teacher_images=len(c['teacher_training_paths']), inference_vlm=False,
                  packed_heads=False, image_size=list(size), original_weight=args.original_weight)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['fit', 'export'])
    parser.add_argument('--checkpoint', required=True, help='fit: WiSE checkpoint; export: output of fit')
    parser.add_argument('--output', required=True)
    parser.add_argument('--cache', help='fit: readout feature cache')
    parser.add_argument('--teacher', default='teacher/targets.npz', help='fit: cached VLM scores')
    parser.add_argument('--original-weight', type=float, default=.25, help='export: weight of the original head')
    args = parser.parse_args()
    torch.set_num_threads(4)
    (fit if args.action == 'fit' else export)(args)
