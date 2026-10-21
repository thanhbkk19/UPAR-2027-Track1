"""Restore verified original weights from the separately supplied organizer bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from common import ROOT, check_hash

ALLOWED = {
    'dist/UPAR_Track1_PE_EVA_Balanced_v18.zip',
    'artifacts/robust_heads_v15/pe_release.pt',
    'artifacts/fast_robust_v17/eva_release.pt',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True, type=Path)
    a = p.parse_args()
    expected = json.loads((ROOT / 'reference/checkpoint_hashes.json').read_text())
    with tarfile.open(a.bundle, 'r:*') as archive:
        entries = archive.getmembers()
        if len({m.name for m in entries}) != len(entries):
            raise ValueError('Duplicate bundle members')
        lookup = {m.name:m for m in entries}
        if not ALLOWED <= lookup.keys():
            raise ValueError('Missing required bundle weights')
        for name in sorted(ALLOWED):
            member = lookup[name]
            if not member.isfile():
                raise ValueError(f'Expected regular file: {name}')
            target = ROOT / name
            if target.exists():
                check_hash(target, expected[name]['sha256'])
                print(f'Already verified: {name}')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + '.tmp')
            try:
                with archive.extractfile(member) as source, temporary.open('wb') as dest:
                    shutil.copyfileobj(source, dest, length=1024*1024)
                check_hash(temporary, expected[name]['sha256'])
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            print(f'Restored and verified: {name}')


if __name__ == '__main__':
    main()
