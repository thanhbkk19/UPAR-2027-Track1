"""Cache the 5 pooled tokens of every train/tune image for frozen-feature readout fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import UPAR
from model import ClipLarge, load_encoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True, help='timm model name')
    p.add_argument('--height', type=int, required=True)
    p.add_argument('--width', type=int, required=True)
    p.add_argument('--weights', help='Pretrained safetensors (frozen-head stage)')
    p.add_argument('--checkpoint', help='Fine-tuned/interpolated checkpoint (readout stage)')
    p.add_argument('--split-root', required=True, help='Directory with train.csv and tune.csv')
    p.add_argument('--output', required=True)
    p.add_argument('--inference-size', type=int, nargs=2, metavar=('H', 'W'), help='Run the encoder at this size')
    p.add_argument('--pure-half', action='store_true', help='FP16 backbone as deployed (default BF16 autocast)')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    manifest = dict(args=vars(args), splits={})
    if args.checkpoint:
        model, state = load_encoder(args.checkpoint)
        assert (state['config']['args']['height'], state['config']['args']['width']) == (args.height, args.width)
        with open(args.checkpoint, 'rb') as f:
            manifest['fine_tuned_checkpoint'] = dict(path=args.checkpoint, sha256=hashlib.file_digest(f, 'sha256').hexdigest(),
                                                     selected_epoch=state['epoch']+1)
        del state
    else:
        model = ClipLarge(args.name, (args.height, args.width), args.weights)
    size = tuple(args.inference_size or (args.height, args.width))
    if args.inference_size:
        model.set_resolution(size)
    model = model.cuda().eval()
    if args.pure_half:
        model.backbone.half()
        model.mean, model.std = model.mean.half(), model.std.half()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'tune']:
        annotation = Path(args.split_root)/f'{split}.csv'
        ds = UPAR(annotation, size)
        file = out/f'{split}_features.npy'
        if file.exists():
            raise FileExistsError(f'Refusing to overwrite {file}')
        features = np.lib.format.open_memmap(file, mode='w+', dtype=np.float16, shape=(len(ds), 5, model.head.dim))
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
        started = time.monotonic()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=not args.pure_half):
            for step, (x, _, ii) in enumerate(loader):
                x = x.cuda(non_blocking=True)
                result = model.features(x.half() if args.pure_half else x).cpu().numpy()
                if not np.isfinite(result).all():
                    raise ValueError('Nonfinite features')
                features[ii.numpy()] = result
                if step % 50 == 0:
                    print(json.dumps(dict(split=split, images=int(ii[-1])+1, total=len(ds))), flush=True)
        features.flush()
        del features
        np.savez_compressed(out/f'{split}_labels.npz', paths=np.array(ds.images), labels=ds.labels, names=ds.names)
        manifest['splits'][split] = dict(images=len(ds), seconds=time.monotonic()-started)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
