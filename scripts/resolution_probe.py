"""Evaluate frozen source-excluded PE/EVA models at cheaper token grids.

Screen first; audit only an explicitly chosen geometry. Each feature pass
produces both the original v11 classifier and the fixed v15 robust classifier.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from clip_large import ClipLarge, RegionHead
from data import resolve_image
from robust_head_screen import SmoothRegionHead


SIZES = {'native': {'pe': (336, 168), 'eva': (224, 224)},
         'small': {'pe': (280, 140), 'eva': (196, 196)},
         'tiny': {'pe': (252, 126), 'eva': (182, 182)}}


def resize_model(model, size):
    """Resample the original positional embedding and rebuild body pooling."""
    patch = model.backbone.patch_embed.patch_size[0]
    if len(size) != 2 or any(d <= 0 or d % patch for d in size):
        raise ValueError(f'Image dimensions must be positive multiples of patch size {patch}')
    model.backbone.set_input_size(img_size=size)
    model.grid = (size[0] // patch, size[1] // patch)
    y = (torch.arange(model.grid[0]) + .5) / model.grid[0]
    x = (torch.arange(model.grid[1]) + .5) / model.grid[1]
    centers = torch.tensor([.13, .40, .75, .5])
    widths = torch.tensor([.15, .24, .25, .60])
    prior = -.5 * ((y[None, :, None] - centers[:, None, None]) / widths[:, None, None]) ** 2
    prior = prior - .5 * ((x[None, None, :] - .5) / .5) ** 2
    model.pool_weights = prior.flatten(1).softmax(-1).to(model.pool_weights)


class JointHeads(torch.nn.Module):
    def __init__(self, model, states, teacher):
        super().__init__()
        self.model = model
        self.heads = torch.nn.ModuleList()
        for state in states:
            head = SmoothRegionHead(model.head.dim, 0)
            head.load_state_dict(state)
            self.heads.append(head)
        self.teacher = RegionHead(True, model.head.dim)
        self.teacher.load_state_dict(teacher['head'])
        self.register_buffer('columns', teacher['columns'])

    def forward(self, pixels):
        x = self.model.features(pixels)
        original = self.model.head(x).float()
        p = torch.stack([h(x).float().sigmoid() for h in self.heads]).mean(0).clamp(1e-7, 1-1e-7)
        z = torch.log(p) - torch.log1p(-p)
        t = self.teacher(x).float().sigmoid().clamp(1e-7, 1-1e-7)
        tz = torch.log(t) - torch.log1p(-t)
        delta = .2 * (tz.index_select(1, self.columns) - z.index_select(1, self.columns))
        robust = z + torch.zeros_like(z).index_copy(1, self.columns, delta)
        return original, robust


class Images(Dataset):
    def __init__(self, paths, size):
        self.paths = [resolve_image(str(p), 'data') for p in paths]
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            a = np.array(image.convert('RGB').resize((self.size[1], self.size[0]), Image.Resampling.BILINEAR))
        x = torch.from_numpy(a).permute(2, 0, 1).contiguous().float().div_(255).sub_(.5).div_(.5)
        return x, index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='artifacts/resolution_v17')
    parser.add_argument('--head-root', default='artifacts/robust_heads_v15')
    parser.add_argument('--stage', choices=['screen', 'audit'], default='screen')
    parser.add_argument('--sizes', nargs='+', choices=list(SIZES), default=list(SIZES))
    parser.add_argument('--folds', nargs='+', default=['PETA', 'Market1501', 'PA100k'])
    parser.add_argument('--members', nargs='+', choices=['pe', 'eva'], default=['pe', 'eva'])
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    for held in args.folds:
        reference = np.load(f'artifacts/loso/wise2_pe_{held}.npz' if args.stage == 'screen'
                            else f'artifacts/robust_heads_v15/audit_{held}.npz')
        if args.stage == 'audit':
            groups = json.loads(Path('data/splits/train90_v1/groups.json').read_text())
            screen = np.load(f'artifacts/loso/wise2_pe_{held}.npz')
            screen_groups = {groups.get(str(p), str(p)) for p in screen['paths']}
            assert screen_groups.isdisjoint({groups.get(str(p), str(p)) for p in reference['paths']})
        for member in args.members:
            manifest = json.loads(Path(f'artifacts/features/foldwise_{member}_{held}/manifest.json').read_text())
            checkpoint = manifest['fine_tuned_checkpoint']['path']
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            cfg = state['config']
            a = cfg['args']
            train_paths = Path(a['split_root']) / 'train.csv'
            # Fold backbone training must exclude the evaluated source.
            assert not any(line.startswith(held + '/') for line in train_paths.read_text().splitlines()[1:])
            heads = torch.load(Path(args.head_root)/f'mixup_{member}_{held}.pt', map_location='cpu', weights_only=False)
            teacher = torch.load(Path(args.head_root)/f'vlm_{member}_{held}.pt', map_location='cpu', weights_only=False)
            assert held not in heads['sources'] and teacher['excluded_source'] == held
            for variant in args.sizes:
                target = root / f'{args.stage}_{variant}_{member}_{held}.npz'
                if target.exists():
                    print(json.dumps({'skip_existing': str(target)}), flush=True)
                    continue
                size = SIZES[variant][member]
                model = ClipLarge(a['tag'], regional=cfg['regional'],
                                  size=(a.get('height', 224), a.get('width', 224)), name=a.get('timm_name'))
                model.load_state_dict(state['ema'], strict=True)
                resize_model(model, size)
                net = JointHeads(model, heads['heads'], teacher).eval().cuda()
                net.model.backbone.half()
                net.model.mean = net.model.mean.half()
                net.model.std = net.model.std.half()
                ds = Images(reference['paths'], size)
                loader = DataLoader(ds, batch_size=args.batch_size, num_workers=6, pin_memory=True)
                base, robust = np.empty_like(reference['labels']), np.empty_like(reference['labels'])
                started = time.monotonic()
                with torch.inference_mode():
                    for pixels, indices in loader:
                        b, r = net(pixels.cuda(non_blocking=True).half())
                        base[indices.numpy()] = b.cpu().numpy()
                        robust[indices.numpy()] = r.cpu().numpy()
                assert np.isfinite(base).all() and np.isfinite(robust).all()
                np.savez_compressed(target, paths=reference['paths'], labels=reference['labels'], base=base, robust=robust)
                report = {'fold': held, 'member': member, 'variant': variant, 'size': size,
                          'checkpoint': checkpoint, 'checkpoint_sha256': manifest['fine_tuned_checkpoint']['sha256'],
                          'stage': args.stage, 'head_root': args.head_root, 'batch_size': args.batch_size,
                          'images': len(ds), 'seconds': time.monotonic()-started,
                          'paths_sha256': hashlib.sha256('\n'.join(reference['paths']).encode()).hexdigest()}
                target.with_suffix('.json').write_text(json.dumps(report, indent=2))
                print(json.dumps(report), flush=True)
                del net, model
                gc.collect()
                torch.cuda.empty_cache()
            del state


if __name__ == '__main__':
    main()
