"""Download the pinned public PE-Core-L and EVA02-L weights (optionally the Qwen3-VL teacher) and verify them."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher', action='store_true', help='Also download Qwen3-VL-8B (only to regenerate teacher scores)')
    a = p.parse_args()
    from huggingface_hub import hf_hub_download, snapshot_download
    models = json.loads((ROOT/'configs/pretrained.json').read_text())
    for name in ['pe', 'eva']+(['teacher'] if a.teacher else []):
        spec = models[name]
        target = ROOT/spec['directory']
        if name == 'teacher':
            snapshot_download(spec['repo_id'], revision=spec['revision'], local_dir=target)
        else:
            path = target/spec['filename']
            if not path.exists():
                hf_hub_download(spec['repo_id'], spec['filename'], revision=spec['revision'], local_dir=target)
            with path.open('rb') as f:
                if hashlib.file_digest(f, 'sha256').hexdigest() != spec['sha256']:
                    raise ValueError(f'Hash mismatch: {path}')
        print(f'{name}: {spec["repo_id"]}@{spec["revision"]} OK')


if __name__ == '__main__':
    main()
