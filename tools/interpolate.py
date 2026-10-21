"""Interpolate the supervised frozen-head initialization and selected EMA."""
import argparse
import sys
from pathlib import Path
from common import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--head', required=True)
    p.add_argument('--alpha', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not 0 <= a.alpha <= 1 or a.output.exists():
        raise ValueError('Alpha must lie in [0,1] and output must not exist')
    import torch
    sys.path.insert(0, str(ROOT / 'scripts'))
    from clip_large import ClipLarge
    torch.set_num_threads(4)
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    args = state['config']['args']
    head = torch.load(a.head, map_location='cpu', weights_only=False)
    model = ClipLarge(args['tag'], a.weights, head['regional'],
                      size=(args['height'], args['width']), name=args['timm_name'])
    model.head.load_state_dict(head['head'], strict=True)
    initial = model.state_dict()
    adapted = state['ema']
    if initial.keys() != adapted.keys():
        raise ValueError('Initialization and EMA architecture differ')
    blended = {k: ((initial[k].float() * (1-a.alpha) + adapted[k].float() * a.alpha).to(initial[k].dtype)
                   if initial[k].is_floating_point() else adapted[k]) for k in initial}
    config = dict(state['config'], wise={'source': a.checkpoint, 'alpha': a.alpha,
                                       'head': a.head, 'weights': a.weights})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'ema': blended, 'config': config, 'epoch': state['epoch']}, a.output)
    print(f'Saved {a.output}; selected EMA epoch {state["epoch"]+1}, alpha={a.alpha}')


if __name__ == '__main__':
    main()
