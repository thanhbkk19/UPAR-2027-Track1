"""Report actual package/device versions and verify the recorded environment."""
import argparse
import importlib.metadata
import json
import platform
from common import ROOT, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--require-cuda', action='store_true')
    p.add_argument('--strict', action='store_true')
    a = p.parse_args()
    packages = {}
    mismatches = []
    for line in (ROOT / 'requirements.txt').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        name, expected = line.split('==')
        actual = importlib.metadata.version(name)
        packages[name] = actual
        if actual.split('+')[0] != expected:
            mismatches.append(f'{name}: expected {expected}, installed {actual}')
    import torch
    cuda = torch.cuda.is_available()
    report = {'python':platform.python_version(),'packages':packages,'cuda_available':cuda,
              'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name() if cuda else None,
              'recorded_training_cuda_runtime':'12.8',
              'cuda_matches_recorded_training':torch.version.cuda == '12.8',
              'mismatches':mismatches}
    if platform.python_version_tuple()[:2] != ('3','12'):
        mismatches.append('Recorded interpreter is Python 3.12')
    write_json(ROOT / 'reports/local/environment.json', report)
    print(json.dumps(report, indent=2))
    if a.require_cuda and not cuda:
        raise SystemExit('CUDA is required for training/export and the FP16 submission runtime')
    if a.strict and mismatches:
        raise SystemExit('Environment differs from the recorded training environment')


if __name__ == '__main__':
    main()
