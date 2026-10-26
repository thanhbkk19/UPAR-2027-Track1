"""Source-excluded (leave-one-source-out) audit: score each fold's exported PE/EVA committee on its held-out source."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ROOT, SOURCES, DeployedImages, read_annotations, source_mean_prior
from metrics import score_dataset, sigmoid


def logit(p):
    return np.log(p)-np.log1p(-p)


@torch.inference_mode()
def predict(model_path, paths, size, batch_size):
    model = torch.jit.load(str(model_path), map_location='cuda').eval()
    out = np.empty((len(paths), 40), dtype=np.float32)
    loader = DataLoader(DeployedImages(paths, size), batch_size=batch_size, num_workers=6, pin_memory=True)
    with torch.jit.optimized_execution(False):
        for pixels, ii in loader:
            out[ii.numpy()] = model(pixels.cuda(non_blocking=True).half()).float().sigmoid().cpu().numpy()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', default=str(ROOT/'work/loso'))
    p.add_argument('--output', default=str(ROOT/'reports/loso_audit.json'))
    p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args()
    report = {}
    for held in SOURCES:
        names, paths, labels = read_annotations(ROOT/f'data/splits/loso_{held}/audit.csv')
        assert all(q.startswith(held+'/') for q in paths)
        work = Path(args.work)/held
        members = [predict(work/'pe_export.pt', paths, (336, 168), args.batch_size),
                   predict(work/'eva_export.pt', paths, (196, 196), args.batch_size)]
        prior = source_mean_prior(ROOT/f'data/splits/loso_{held}/train.csv')
        decision = sigmoid(logit(np.mean(members, 0).clip(1e-7, 1-1e-7))-.4*logit(prior))
        report[held] = dict(images=len(paths), **score_dataset(labels, decision))
        np.savez_compressed(work/'audit_predictions.npz', paths=np.array(paths), labels=labels, probabilities=decision)
        print(json.dumps({held: {k: round(report[held][k], 6) for k in ['mA', 'Instance_F1', 'Challenge_Avg']}}), flush=True)
    report['mean'] = {k: float(np.mean([report[s][k] for s in SOURCES])) for k in ['mA', 'Instance_F1', 'Challenge_Avg']}
    report['images'] = sum(report[s]['images'] for s in SOURCES)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(mean=report['mean'], images=report['images'])))


if __name__ == '__main__':
    main()
