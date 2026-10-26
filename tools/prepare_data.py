"""Rebuild every split CSV from the official annotations and the recorded path lists; link the image folders."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ['PETA', 'Market1501', 'PA100k']


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_rows(path):
    with Path(path).open(newline='') as f:
        rows = list(csv.reader(f))
    return rows[0][1:], {r[0]: r[1:] for r in rows[1:]}


def csv_bytes(names, paths, rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['image']+names)
    for path in paths:
        writer.writerow([path]+[int(float(v)) for v in rows[path]])
    return stream.getvalue().encode()


def path_list(name):
    with gzip.open(ROOT/f'splits/{name}.txt.gz', 'rt') as f:
        return f.read().splitlines()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--official-train', required=True, type=Path, help='starter kit annotations/task1/train/gt.csv')
    p.add_argument('--official-val', required=True, type=Path, help='starter kit annotations/task1/val/gt.csv')
    p.add_argument('--images-root', required=True, type=Path, help='Folder containing PETA/, Market1501/, PA100k/')
    a = p.parse_args()
    hashes = json.loads((ROOT/'reference/hashes.json').read_text())
    for path, key in [(a.official_train, 'official_train'), (a.official_val, 'official_val')]:
        if sha256(path) != hashes[key]:
            raise ValueError(f'{path} is not the official UPAR 2027 Track 1 annotation file')
    names, rows = load_rows(a.official_train)
    val_names, val_rows = load_rows(a.official_val)
    assert names == val_names and not rows.keys() & val_rows.keys()
    rows |= val_rows

    targets = {f'release/{s}': (path_list(s), f'data/splits/release/{s}.csv') for s in ['train', 'tune', 'audit', 'val']}
    targets['teacher'] = (path_list('teacher'), 'data/splits/teacher.csv')
    for fold in FOLDS:
        for s in ['train', 'tune', 'audit']:
            targets[f'loso_{fold}/{s}'] = (path_list(f'loso_{fold}/{s}'), f'data/splits/loso_{fold}/{s}.csv')
    for key, (paths, target) in targets.items():
        payload = csv_bytes(names, paths, rows)
        if key in hashes['csv'] and hashlib.sha256(payload).hexdigest() != hashes['csv'][key]:
            raise ValueError(f'Reconstructed split differs from the recorded one: {key}')
        (ROOT/target).parent.mkdir(parents=True, exist_ok=True)
        (ROOT/target).write_bytes(payload)

    # Sanity checks: release train/tune/audit are disjoint, LOSO folds never train on the held-out source.
    release = {s: set(targets[f'release/{s}'][0]) for s in ['train', 'tune', 'audit']}
    assert not (release['train'] & release['tune'] or release['train'] & release['audit'] or release['tune'] & release['audit'])
    for fold in FOLDS:
        train = targets[f'loso_{fold}/train'][0]
        assert not any(q.startswith(fold+'/') for q in train)
        assert all(q.startswith(fold+'/') for q in targets[f'loso_{fold}/audit'][0])

    for source in FOLDS:
        origin, link = (a.images_root/source).resolve(), ROOT/'data'/source
        if not origin.is_dir():
            raise FileNotFoundError(origin)
        if not link.exists():
            link.symlink_to(origin, target_is_directory=True)
    import sys
    sys.path.insert(0, str(ROOT/'scripts'))
    from data import resolve_image
    for path in rows:
        resolve_image(path, ROOT/'data')
    print(json.dumps({key: len(paths) for key, (paths, _) in targets.items()}, indent=1))
    print('All split CSVs rebuilt and all images found.')


if __name__ == '__main__':
    main()
