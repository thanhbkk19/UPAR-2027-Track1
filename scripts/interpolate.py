"""WiSE weight interpolation: alpha * fine-tuned EMA + (1 - alpha) * (pretrained encoder + frozen head)."""
import argparse
from pathlib import Path

import torch

from model import ClipLarge


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, help='best.pt or last.pt from finetune.py')
    p.add_argument('--weights', required=True, help='Pretrained safetensors')
    p.add_argument('--head', required=True, help='Frozen-feature head used to initialise fine-tuning')
    p.add_argument('--alpha', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not 0 <= a.alpha <= 1 or a.output.exists():
        raise ValueError('Alpha must lie in [0, 1] and the output must not exist')
    torch.set_num_threads(4)
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    args = state['config']['args']
    model = ClipLarge(args['timm_name'], (args['height'], args['width']), a.weights)
    model.head.load_state_dict(torch.load(a.head, map_location='cpu', weights_only=False)['head'], strict=True)
    initial, adapted = model.state_dict(), state['ema']
    if initial.keys() != adapted.keys():
        raise ValueError('Initialisation and EMA architectures differ')
    blended = {k: ((initial[k].float()*(1-a.alpha)+adapted[k].float()*a.alpha).to(initial[k].dtype)
                   if initial[k].is_floating_point() else adapted[k]) for k in initial}
    config = dict(state['config'], wise=dict(source=a.checkpoint, alpha=a.alpha, head=a.head, weights=a.weights))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(ema=blended, config=config, epoch=state['epoch']), a.output)
    print(f'Saved {a.output}: EMA epoch {state["epoch"]+1}, alpha {a.alpha}')


if __name__ == '__main__':
    main()
