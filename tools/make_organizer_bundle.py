"""Package this source tree plus the three original reference artifacts."""
import argparse
import subprocess
import tarfile
from pathlib import Path
from common import ROOT, check_hash, read_json, sha256, write_json
from restore_bundle import ALLOWED


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts-root', type=Path, required=True,
                   help='Directory containing dist/ and artifacts/ for the original release')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    expected = read_json(ROOT / 'reference/checkpoint_hashes.json')
    for name in sorted(ALLOWED):
        check_hash(a.artifacts_root / name, expected[name]['sha256'])
    # Git's index is an allowlist: never sweep the workspace, private keys,
    # datasets, local logs, caches or Python environments into the handoff.
    tracked = subprocess.check_output(['git','ls-files','-z'], cwd=ROOT).decode().split('\0')
    files = [name for name in tracked if name]
    if not files:
        raise ValueError('Stage/commit the source files before making a handoff bundle')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(a.output, 'w') as archive:
        for name in files:
            archive.add(ROOT / name, arcname='source/' + name, recursive=False)
        for name in sorted(ALLOWED):
            archive.add((a.artifacts_root / name).resolve(), arcname=name, recursive=False)
    digest = sha256(a.output)
    a.output.with_suffix(a.output.suffix+'.sha256').write_text(f'{digest}  {a.output.name}\n')
    write_json(a.output.with_suffix('.contents.json'),
               {'sha256':digest,'source_files':len(files),'reference_artifacts':expected,
                'note':'Original images and generic foundation weights are not redistributed.'})
    print(f'Created {a.output} ({a.output.stat().st_size/1e9:.3f} GB), SHA256 {digest}')


if __name__ == '__main__':
    main()
