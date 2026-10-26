"""Run the training pipeline stage by stage.

  release : the submitted predictor (both encoders trained on all three sources) -> work/release/submission.zip
  loso    : the paper's source-excluded audit (one PE/EVA pair per held-out source) -> reports/loso_audit.json

A stage is skipped when work/done/<stage> exists, so an interrupted run continues with --run again.
"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ['PETA', 'Market1501', 'PA100k']
ENCODERS = {
    'pe': dict(name='vit_pe_core_large_patch14_336.fb', size=(336, 168), weights='pretrained/pe_core_l/model.safetensors',
               alpha=.8, loso_alpha=.8, readout=['--batch-size', '256', '--workers', '12']),
    'eva': dict(name='eva02_large_patch14_clip_224.merged2b', size=(224, 224), weights='pretrained/eva02_l_clip/model.safetensors',
                alpha=.75, loso_alpha=.7,
                # Readouts are fitted on the features the deployed model sees: 196x196, pure FP16.
                readout=['--batch-size', '128', '--workers', '6', '--inference-size', '196', '196', '--pure-half']),
}


def encoder_stages(e, split, work, frozen_args, select, alpha, tag, micro_batches=1):
    c = ENCODERS[e]
    h, w = map(str, c['size'])
    shape = ['--height', h, '--width', w]
    return [
        (f'frozen-{e}{tag}', ['scripts/fit_frozen_head.py', '--cache', f'work/features/{e}_pretrained',
                              '--output', f'{work}/{e}_frozen', *frozen_args]),
        (f'finetune-{e}{tag}', ['scripts/finetune.py', '--weights', c['weights'], '--head-checkpoint', f'{work}/{e}_frozen/head.pt',
                                '--timm-name', c['name'], *shape, '--split-root', split, '--output', f'{work}/{e}_finetune',
                                '--micro-batches', str(micro_batches)]),
        (f'wise-{e}{tag}', ['scripts/interpolate.py', '--checkpoint', f'{work}/{e}_finetune/{select}.pt', '--weights', c['weights'],
                            '--head', f'{work}/{e}_frozen/head.pt', '--alpha', str(alpha), '--output', f'{work}/{e}_wise.pt']),
        (f'readout-cache-{e}{tag}', ['scripts/cache_features.py', '--name', c['name'], *shape, '--checkpoint', f'{work}/{e}_wise.pt',
                                     '--split-root', split, '--output', f'work/features/{e}_readout{tag}', *c['readout']]),
        (f'readouts-{e}{tag}', ['scripts/readouts.py', 'fit', '--cache', f'work/features/{e}_readout{tag}',
                                '--checkpoint', f'{work}/{e}_wise.pt', '--output', f'{work}/{e}_committee.pt']),
        (f'export-{e}{tag}', ['scripts/readouts.py', 'export', '--checkpoint', f'{work}/{e}_committee.pt',
                              '--output', f'{work}/{e}_export.pt']),
    ]


def stages(target, micro_batches=1):
    out = []
    for e, c in ENCODERS.items():
        h, w = map(str, c['size'])
        out.append((f'cache-{e}', ['scripts/cache_features.py', '--name', c['name'], '--height', h, '--width', w,
                                   '--weights', c['weights'], '--split-root', 'data/splits/release',
                                   '--output', f'work/features/{e}_pretrained']))
    if target in ('release', 'all'):
        for e in ENCODERS:
            # Frozen head: 25 epochs, inverse-sqrt source weighting, best tune epoch; encoder: best EMA epoch.
            out += encoder_stages(e, 'data/splits/release', 'work/release', [], 'best', ENCODERS[e]['alpha'], '', micro_batches)
        out += [('package', ['scripts/package.py', '--pe', 'work/release/pe_export.pt', '--eva', 'work/release/eva_export.pt',
                             '--output', 'work/release/submission.zip']),
                ('evaluate', ['scripts/evaluate_archive.py', '--archive', 'work/release/submission.zip',
                              '--output', 'reports/release_audit.json'])]
    if target in ('loso', 'all'):
        for fold in FOLDS:
            for e in ENCODERS:
                # LOSO fold recipe: frozen head on the two training sources (12 epochs, last), last EMA epoch.
                frozen = ['--exclude-source', fold, '--epochs', '12', '--domain-power', '0', '--select', 'last',
                          '--clip-grad', '0', '--seed', '1']
                out += encoder_stages(e, f'data/splits/loso_{fold}', f'work/loso/{fold}', frozen, 'last',
                                      ENCODERS[e]['loso_alpha'], f'-{fold}', micro_batches)
        out.append(('evaluate-loso', ['scripts/evaluate_loso.py']))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('target', choices=['release', 'loso', 'all'])
    p.add_argument('--run', action='store_true', help='Execute (default: print the plan)')
    p.add_argument('--only', nargs='+', help='Run only these stages')
    p.add_argument('--micro-batches', type=int, default=1, help='Fine-tuning memory saver (same batch of 64)')
    a = p.parse_args()
    plan = stages(a.target, a.micro_batches)
    if a.only:
        unknown = set(a.only)-{n for n, _ in plan}
        if unknown:
            p.error(f'Unknown stages: {sorted(unknown)}')
        plan = [(n, c) for n, c in plan if n in a.only]
    for name, command in plan:
        done = ROOT/'work/done'/name
        print(f'{"[done] " if done.exists() else ""}{name}: python {" ".join(command)}', flush=True)
        if not a.run or done.exists():
            continue
        log = ROOT/'logs'/f'{name}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w') as stream:
            result = subprocess.run([sys.executable, '-u', *command], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            sys.exit(f'{name} failed; see {log}')
        done.parent.mkdir(parents=True, exist_ok=True)
        done.touch()
        print(f'  -> finished ({log.relative_to(ROOT)})', flush=True)


if __name__ == '__main__':
    main()
