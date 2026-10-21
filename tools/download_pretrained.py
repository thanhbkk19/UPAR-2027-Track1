"""Download pinned public foundation models; verify encoder weight hashes."""
import argparse
from common import ROOT, check_hash, read_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher', action='store_true', help='Also download optional Qwen teacher')
    a = p.parse_args()
    from huggingface_hub import hf_hub_download, snapshot_download
    models = read_json(ROOT / 'configs/pretrained.json')
    for name in ['pe', 'eva'] + (['teacher'] if a.teacher else []):
        spec = models[name]
        target = ROOT / spec['directory']
        if name == 'teacher':
            snapshot_download(spec['repo_id'], revision=spec['revision'], local_dir=target)
        else:
            path = target / spec['filename']
            if not path.exists():
                hf_hub_download(spec['repo_id'], spec['filename'], revision=spec['revision'], local_dir=target)
            check_hash(path, spec['sha256'])
        print(f'{name}: verified pinned source {spec["repo_id"]}@{spec["revision"]}')


if __name__ == '__main__':
    main()
