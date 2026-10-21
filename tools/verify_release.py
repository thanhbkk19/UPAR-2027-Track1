"""Verify archive integrity and the complete Balanced inference configuration."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from common import ROOT, read_json, sha256, write_json

REFERENCE_SHA256 = '3cb753cf4eaedc93885459e391cccd5d091b98ec7cbd31214cd95abf86d49d9d'


def verify(path, require_reference=False):
    reference = read_json(ROOT / 'reference/submission_config.json')
    reference_members = read_json(ROOT / 'reference/submission_manifest.json')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError('Duplicate archive members')
        if set(names) != set(reference_members) | {'manifest.json'}:
            raise ValueError('Unexpected or missing submission members')
        if any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):
            raise ValueError('Unsafe archive member')
        manifest = json.loads(archive.read('manifest.json'))
        if set(manifest) != set(reference_members):
            raise ValueError('Incomplete hash manifest')
        for name, digest in manifest.items():
            with archive.open(name) as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != digest:
                raise ValueError(f'Corrupt payload: {name}')
        config = json.loads(archive.read('assets/config.json'))
        for key in ['backend', 'attribute_names', 'logit_offset', 'gpu_microbatch']:
            if config[key] != reference[key]:
                raise ValueError(f'Wrong inference configuration: {key}')
        if len(config['models']) != 2:
            raise ValueError('Expected two encoders')
        for model, expected in zip(config['models'], reference['models']):
            for key in ['file','weight','image_size','flip_tta','amp','half','logit_offset']:
                if model[key] != expected[key]:
                    raise ValueError(f'Wrong model setting: {key}')
            if manifest[model['file']] != model['source_sha256']:
                raise ValueError('Model hash differs from configuration')
        if manifest['run.py'] != reference_members['run.py']:
            raise ValueError('Inference runtime differs from the submitted runtime')
    digest = sha256(path)
    if require_reference and digest != REFERENCE_SHA256:
        raise ValueError('This is not the original submitted archive')
    return {'status':'PASS','archive_sha256':digest,
            'is_original_reference_archive':digest == REFERENCE_SHA256,
            'runtime_matches_reference':True,'inference_configuration_matches_reference':True,
            'model_files_match_reference':all(manifest[m['file']] == reference_members[m['file']] for m in config['models']),
            'scope':'Structural/hash verification; run validate_submission.py for inference and metrics.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', required=True, type=Path)
    p.add_argument('--require-reference', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT / 'reports/local/archive_integrity.json')
    a = p.parse_args()
    result = verify(a.archive, a.require_reference)
    write_json(a.output, result)
    print(json.dumps(result, indent=2))
