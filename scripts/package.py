"""Build the offline organizer ZIP (runtime/run.py + two TorchScript encoders + config)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

from data import ROOT, source_mean_prior

RUNTIME = ['run.py', 'metadata.yaml', 'NOTICE.md', 'LICENSE-PE.txt', 'LICENSE-EVA.txt']


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pe', type=Path, required=True)
    p.add_argument('--eva', type=Path, required=True)
    p.add_argument('--train-csv', default=str(ROOT/'data/splits/release/train.csv'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    config = json.loads((ROOT/'reference/submission_config.json').read_text())
    # Decision rule: subtract 0.4 * logit(training prior). The released values carry float32 rounding
    # from an earlier bias add/subtract; they are kept after checking them against the prior.
    prior = source_mean_prior(a.train_csv)
    np.testing.assert_allclose(.4*(np.log(prior)-np.log1p(-prior)), config['logit_offset'], rtol=0, atol=3e-7)
    models = {}
    for path, entry, size in zip([a.pe, a.eva], config['models'], [[336, 168], [196, 196]]):
        export = json.loads(path.with_suffix('.json').read_text())
        if export['image_size'] != size or export['original_weight'] != .25:
            raise ValueError(f'Export does not match the release recipe: {path}')
        if export['sources'] != ['Market1501', 'PA100k', 'PETA']:
            raise ValueError('The release committee needs all three training sources')
        entry['source_sha256'] = sha256(path)
        entry['robust_committee']['export'] = export
        models[entry['file']] = path
    payload = {name: ROOT/'runtime'/name for name in RUNTIME} | models
    config_bytes = (json.dumps(config, indent=2)+'\n').encode()
    manifest = {name: sha256(path) for name, path in payload.items()}
    manifest['assets/config.json'] = hashlib.sha256(config_bytes).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.output, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name, path in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 17, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with path.open('rb') as source, archive.open(info, 'w', force_zip64=True) as dest:
                shutil.copyfileobj(source, dest, length=1 << 20)
        archive.writestr('assets/config.json', config_bytes)
        archive.writestr('manifest.json', json.dumps(manifest, indent=2))
    print(json.dumps(dict(output=str(a.output), sha256=sha256(a.output))))


if __name__ == '__main__':
    main()
