"""Compare audit predictions (evaluate_archive.py .npz) with those of the submitted best_submission.zip."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate', type=Path)
    p.add_argument('--require-exact', action='store_true')
    a = p.parse_args()
    reference = ROOT/'reference/audit_probabilities.npz'
    with reference.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == json.loads((ROOT/'reference/hashes.json').read_text())['reference_audit_probabilities']
    with np.load(reference) as old, np.load(a.candidate) as new:
        if not np.array_equal(old['paths'], new['paths']):
            raise ValueError('Evaluate the full release audit in its original row order')
        delta = np.abs(old['probabilities']-new['probabilities'])
        report = dict(images=len(delta), max_probability_difference=float(delta.max()),
                      changed_decisions=int(((old['probabilities'] > .5) != (new['probabilities'] > .5)).sum()),
                      exact_match=bool(np.array_equal(old['probabilities'], new['probabilities'])))
    print(json.dumps(report, indent=2))
    if a.require_exact and not report['exact_match']:
        raise SystemExit('Predictions differ from the submitted archive')


if __name__ == '__main__':
    main()
