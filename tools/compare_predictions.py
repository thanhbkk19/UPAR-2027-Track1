"""Compare evaluator predictions with the original archive on identical images."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import ROOT, check_hash, read_json, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', required=True, type=Path)
    p.add_argument('--output', type=Path, default=ROOT/'reports/local/prediction_comparison.json')
    p.add_argument('--require-exact', action='store_true', help='Useful for original-checkpoint re-export only')
    a = p.parse_args()
    path = ROOT/'reference/audit_probabilities.npz'
    check_hash(path, read_json(ROOT/'reference/input_hashes.json')['reference_predictions']['sha256'])
    with np.load(path) as reference, np.load(a.candidate) as candidate:
        if not np.array_equal(reference['paths'], candidate['paths']):
            raise ValueError('Use the full in-domain audit in the original row order')
        old, new = reference['probabilities'], candidate['probabilities']
        if old.shape != new.shape or not np.isfinite(new).all():
            raise ValueError('Invalid candidate predictions')
        delta = np.abs(old-new)
        report = {'images':len(old),'max_probability_difference':float(delta.max()),
                  'mean_probability_difference':float(delta.mean()),
                  'changed_binary_predictions':int(((old>.5)!=(new>.5)).sum()),
                  'paths_identical':True,'exact_prediction_match':bool(np.array_equal(old,new))}
    write_json(a.output, report)
    print(json.dumps(report,indent=2))
    if a.require_exact and not report['exact_prediction_match']:
        raise SystemExit('Predictions differ from the original artifact; inspect the saved comparison')


if __name__ == '__main__':
    main()
