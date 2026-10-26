"""Fine-tune the encoder and regional head end to end, keeping an EMA copy."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler

from data import UPAR
from losses import polarity_weights
from metrics import score, sigmoid
from model import ClipLarge


class EpochSampler(Sampler):
    """Seeded permutation per epoch; `start` skips already-seen batches when resuming."""
    def __init__(self, length, seed):
        self.length, self.seed, self.epoch, self.start = length, seed, 0, 0

    def __iter__(self):
        order = torch.randperm(self.length, generator=torch.Generator().manual_seed(self.seed+self.epoch)).tolist()
        return iter(order[self.start:])

    def __len__(self):
        return self.length-self.start


def save(state, path):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    pred = np.empty_like(loader.dataset.labels)
    for x, _, ii in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x.cuda(non_blocking=True))
        pred[ii.numpy()] = logits.float().cpu().numpy()
    return pred, score(loader.dataset.labels, sigmoid(pred), loader.dataset.domains)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', required=True, help='Pretrained safetensors')
    p.add_argument('--head-checkpoint', required=True, help='Frozen-feature head (fit_frozen_head.py)')
    p.add_argument('--timm-name', required=True)
    p.add_argument('--height', type=int, required=True)
    p.add_argument('--width', type=int, required=True)
    p.add_argument('--split-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--last-blocks', type=int, default=24, help='Number of trainable transformer blocks')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--encoder-lr', type=float, default=1e-5)
    p.add_argument('--head-lr', type=float, default=2e-4)
    p.add_argument('--domain-power', type=float, default=.5, help='Row weight = source size ** -power')
    p.add_argument('--seed', type=int, default=2093)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--micro-batches', type=int, default=1, help='Split each batch into chunks to save GPU memory')
    p.add_argument('--resume', help='resume.pt of an interrupted run')
    args = p.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'config.json').exists() and not args.resume:
        raise FileExistsError(f'Run already exists: {out}')

    model = ClipLarge(args.timm_name, (args.height, args.width), args.weights)
    model.head.load_state_dict(torch.load(args.head_checkpoint, map_location='cpu', weights_only=False)['head'])
    model.backbone.requires_grad_(False)
    for block in model.backbone.blocks[-args.last_blocks:]:
        block.requires_grad_(True)
    model.backbone.norm.requires_grad_(True)
    model.cuda()
    ema = copy.deepcopy(model).eval()
    ema.requires_grad_(False)

    root = Path(args.split_root)
    train = UPAR(root/'train.csv', (args.height, args.width), train=True)
    val = UPAR(root/'tune.csv', (args.height, args.width))
    assert train.names == val.names and not set(train.images) & set(val.images)
    sampler = EpochSampler(len(train), args.seed)
    tl = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                    pin_memory=True, persistent_workers=args.workers > 0)
    vl = DataLoader(val, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                    persistent_workers=args.workers > 0)
    steps_per_epoch = math.ceil(len(train)/args.batch_size)

    # Inverse-square-root source weighting (power 0.5) of every row's loss and of the label rates.
    counts = {d: int((train.domains == d).sum()) for d in np.unique(train.domains)}
    dw = np.array([counts[d]**-args.domain_power for d in train.domains], dtype=np.float32)
    dw /= dw.mean()
    rate = torch.tensor(np.average(train.labels, axis=0, weights=dw), device='cuda', dtype=torch.float32)
    positive_weight, negative_weight = polarity_weights(rate)
    domain_weights = torch.from_numpy(dw).cuda()
    encoder = [v for v in model.backbone.parameters() if v.requires_grad]
    optim = torch.optim.AdamW([dict(params=encoder, lr=args.encoder_lr, base_lr=args.encoder_lr),
                               dict(params=list(model.head.parameters()), lr=args.head_lr, base_lr=args.head_lr)],
                              weight_decay=.01)
    config = dict(args=vars(args), attributes=train.names, train_images=len(train), tune_images=len(val),
                  domain_counts=counts, torch=torch.__version__, gpu=torch.cuda.get_device_name())
    (out/'config.json').write_text(json.dumps(config, indent=2))

    def checkpoint(epoch, **extra):
        return dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optim.state_dict(), epoch=epoch,
                    best=best, global_step=global_step, config=config, **extra)

    best, global_step, start_epoch, start_batch, partial_loss = -1, 0, 0, 0, 0.
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        ema.load_state_dict(state['ema'])
        optim.load_state_dict(state['optimizer'])
        best, global_step, start_epoch = state['best'], state['global_step'], state['epoch']+1
        start_batch, partial_loss = state.get('next_batch', 0), state.get('partial_loss', 0.)
    else:
        logits, metrics = evaluate(ema, vl)
        best = metrics['mean']['Challenge_Avg']
        save(checkpoint(-1), out/'best.pt')
        print(json.dumps(dict(epoch=0, tune_score=best)), flush=True)

    total = args.epochs*steps_per_epoch
    for epoch in range(start_epoch, args.epochs):
        started = time.monotonic()
        model.train()
        loss_sum = partial_loss
        sampler.epoch, sampler.start = epoch, start_batch*args.batch_size
        for block in model.backbone.blocks[:-args.last_blocks]:
            block.eval()
        for step, (x, y, ii) in enumerate(tl, start=start_batch):
            # Linear warm-up over half an epoch, then cosine decay to 5%.
            warmup = min(1., (global_step+1)/(steps_per_epoch*.5))
            progress = max(0., (global_step-steps_per_epoch*.5)/max(total-steps_per_epoch*.5, 1))
            scale = warmup*(.05+.95*.5*(1+math.cos(math.pi*progress)))
            for group in optim.param_groups:
                group['lr'] = group['base_lr']*scale
            x, y, ii = x.cuda(non_blocking=True), y.cuda(non_blocking=True), ii.cuda(non_blocking=True)
            optim.zero_grad(set_to_none=True)
            loss = 0.
            for xc, yc, ic in zip(*(t.tensor_split(args.micro_batches) for t in (x, y, ii))):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    weights = yc*positive_weight+(1-yc)*negative_weight
                    row_loss = (F.binary_cross_entropy_with_logits(model(xc).float(), yc, reduction='none')*weights).mean(1)
                    chunk = (row_loss*domain_weights[ic]).mean()*len(xc)/len(x)
                if not torch.isfinite(chunk):
                    raise ValueError('Nonfinite loss')
                chunk.backward()
                loss += chunk.detach()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optim.step()
            with torch.no_grad():
                decay = min(.999, (1+global_step)/(10+global_step))
                for ep, mp in zip(ema.parameters(), model.parameters()):
                    if mp.requires_grad:
                        ep.lerp_(mp, 1-decay)
            global_step += 1
            loss_sum += loss.item()
            if step % 100 == 0:
                print(json.dumps(dict(epoch=epoch+1, step=step, steps=steps_per_epoch, loss=loss.item(),
                                      seconds=time.monotonic()-started)), flush=True)
            if (step+1) % 200 == 0:
                save(checkpoint(epoch-1, next_batch=step+1, partial_loss=loss_sum), out/'resume.pt')
        logits, metrics = evaluate(ema, vl)
        value = metrics['mean']['Challenge_Avg']
        improved, best = value > best, max(best, value)
        result = dict(epoch=epoch+1, loss=loss_sum/steps_per_epoch, seconds=time.monotonic()-started, **metrics)
        with (out/'history.jsonl').open('a') as f:
            f.write(json.dumps(result)+'\n')
        print(json.dumps(dict(epoch=epoch+1, tune_score=value, seconds=result['seconds'])), flush=True)
        state = checkpoint(epoch)
        save(state, out/'last.pt')
        save(state, out/'resume.pt')
        if improved:
            save(state, out/'best.pt')
        start_batch, partial_loss = 0, 0.
    (out/'complete.json').write_text(json.dumps(dict(epochs=args.epochs, best=best), indent=2))


if __name__ == '__main__':
    main()
