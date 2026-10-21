"""Build the offline organizer ZIP from exported PE/EVA TorchScript models."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from common import ROOT, check_hash, read_json, sha256, write_json


def prior_offsets(csv_path):
    import numpy as np
    sys.path.insert(0, str(ROOT / 'scripts'))
    from data import read_annotations
    _, paths, y = read_annotations('train', csv_path)
    domains = np.array([p.split('/')[0] for p in paths])
    prior = np.mean([y[domains == d].mean(0) for d in np.unique(domains)], axis=0).clip(1e-4, 1-1e-4)
    return .4 * (np.log(prior) - np.log1p(-prior))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pe', type=Path, required=True)
    p.add_argument('--eva', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(f'Choose a fresh output: {a.output}')
    import numpy as np
    config = read_json(ROOT / 'reference/submission_config.json')
    check_hash(ROOT / 'data/splits/train90_v1/train.csv',
               read_json(ROOT / 'splits/report.json')['partitions']['train']['sha256'])
    # Preserve the released float32 offsets, including the historical subtract/
    # add-bias rounding. Validate them independently against training priors.
    np.testing.assert_allclose(prior_offsets(ROOT / 'data/splits/train90_v1/train.csv'),
                               config['logit_offset'], rtol=0, atol=3e-7)
    models = {}
    for source, entry, geometry in zip([a.pe, a.eva], config['models'], [[336,168], [196,196]]):
        export = read_json(source.with_suffix('.json'))
        if export['image_size'] != geometry or export['original_weight'] != .25 or export['packed_heads']:
            raise ValueError(f'Export does not match the Balanced recipe: {source}')
        if export['sources'] != ['Market1501', 'PA100k', 'PETA'] or export['teacher_images'] != 24000:
            raise ValueError('Wrong training source committee or teacher subset')
        entry['source_sha256'] = sha256(source)
        entry['robust_committee']['export'] = export
        entry['robust_committee']['checkpoint'] = export['checkpoint']
        # Metadata comes from this run, not historical epoch/score claims.
        import torch
        state = torch.load(export['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
        entry['training_recipe'] = {'args': state['config']['args'],
                                   'attributes': state['config']['attributes'],
                                   'selected_epoch': state['epoch'] + 1}
        entry['robust_committee']['feature_cache'] = state['robust_committee']['cache']
        del state
        models[entry['file']] = source
    config['reproduction'] = {'target': 'UPAR_Track1_PE_EVA_Balanced_v18.zip',
                              'note': 'Rebuilt from the recorded recipe; evaluate this archive independently.'}
    # The reference selection describes why the fixed recipe was chosen, not
    # measured performance of a fresh retraining run.
    config['robust_selection']['selection'] += ' (historical selection record; not a result of this rebuild)'
    runtime_files = ['run.py', 'metadata.yaml', 'NOTICE.md', 'LICENSE-PE.txt', 'LICENSE-EVA.txt']
    payload = {name: ROOT / 'runtime' / name for name in runtime_files} | models
    config_bytes = (json.dumps(config, indent=2) + '\n').encode()
    manifest = {name: sha256(path) for name, path in payload.items()}
    manifest['assets/config.json'] = hashlib.sha256(config_bytes).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.output.with_suffix('.zip.tmp')
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for name, path in sorted(payload.items()):
                info = zipfile.ZipInfo(name, date_time=(2026,10,17,0,0,0))
                info.compress_type = zipfile.ZIP_DEFLATED
                with path.open('rb') as source, archive.open(info, 'w', force_zip64=True) as dest:
                    import shutil
                    shutil.copyfileobj(source, dest, length=1024*1024)
            archive.writestr('assets/config.json', config_bytes)
            archive.writestr('manifest.json', json.dumps(manifest, indent=2))
        temporary.replace(a.output)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(a.output.with_suffix('.build.json'), {'sha256': sha256(a.output), 'manifest': manifest})
    print(f'Created {a.output}')


if __name__ == '__main__':
    main()
