"""Run the recorded release recipe with explicit stages and verifiable receipts."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from common import ROOT, check_hash, read_json, sha256, write_json


def code_digest():
    files = sorted(p for folder in ['scripts', 'tools', 'runtime', 'configs']
                   for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    return hashlib.sha256(''.join(str(p.relative_to(ROOT)) + sha256(p) for p in files).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true', help='Execute; default only prints commands')
    p.add_argument('--plan', action='store_true', help='Explicitly print commands without running')
    p.add_argument('--from-stage')
    p.add_argument('--to-stage')
    p.add_argument('--stage', help='Run one named stage')
    p.add_argument('--resume', action='store_true', help='Skip only stages whose receipts and outputs still match')
    a = p.parse_args()
    if a.run and a.plan:
        p.error('Choose --run or --plan')
    stages = read_json(ROOT / 'configs/balanced.json')['stages']
    names = [s['name'] for s in stages]
    for name in [a.stage, a.from_stage, a.to_stage]:
        if name and name not in names:
            p.error(f'Unknown stage {name}; choose from {names}')
    if a.stage and (a.from_stage or a.to_stage):
        p.error('--stage cannot be combined with a stage range')
    start = names.index(a.stage or a.from_stage) if a.stage or a.from_stage else 0
    stop = names.index(a.stage or a.to_stage) + 1 if a.stage or a.to_stage else len(stages)
    if stop <= start:
        p.error('Stage range is reversed')
    selected = stages[start:stop]
    for stage in selected:
        print(f'{stage["name"]}: {shlex.join([sys.executable] + stage["command"])}', flush=True)
    if not a.run:
        print('Plan only. Add --run to execute in the repository root.')
        return
    os.chdir(ROOT)
    # The default training path uses the exact cached teacher targets. This also
    # keeps regeneration in a separate environment from backbone training.
    teacher_hash = read_json(ROOT / 'reference/input_hashes.json')['teacher']['sha256']
    check_hash(ROOT / 'teacher/targets.npz', teacher_hash)
    target = ROOT / 'artifacts/loso/vlm_train24k.npz'
    if target.exists():
        check_hash(target, teacher_hash)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'teacher/targets.npz', target)
    for split, spec in read_json(ROOT / 'splits/report.json')['partitions'].items():
        check_hash(ROOT / f'data/splits/train90_v1/{split}.csv', spec['sha256'])
    code = code_digest()
    for stage in selected:
        receipt = ROOT / f'artifacts/pipeline/{stage["name"]}.json'
        inputs = {name: sha256(ROOT / name) for name in stage['inputs']}
        signature = {'command':stage['command'], 'code_sha256':code, 'inputs':inputs}
        if receipt.exists() and a.resume:
            saved = read_json(receipt)
            if saved['signature'] != signature:
                raise ValueError(f'Inputs/code changed for {stage["name"]}; use a fresh checkout for a new run')
            for name, digest in saved['outputs'].items():
                check_hash(ROOT / name, digest)
            print(f'{stage["name"]}: verified existing receipt; skipped', flush=True)
            continue
        exists = [name for name in stage['outputs'] if (ROOT / name).exists()]
        if exists:
            raise FileExistsError(f'{stage["name"]}: outputs exist without a verified resume: {exists}. '
                                  'See README recovery instructions; no outputs were replaced.')
        log = ROOT / f'logs/{stage["name"]}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f'{stage["name"]}: running; log={log.relative_to(ROOT)}', flush=True)
        with log.open('w') as stream:
            subprocess.run([sys.executable, '-u', *stage['command']], cwd=ROOT,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        outputs = {name:sha256(ROOT / name) for name in stage['outputs']}
        write_json(receipt, {'signature':signature,'outputs':outputs,
                            'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        print(f'{stage["name"]}: COMPLETE', flush=True)


if __name__ == '__main__':
    main()
