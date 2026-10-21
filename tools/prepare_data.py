"""Reconstruct the exact recorded CSVs from organizer annotations and path lists."""
import argparse
import csv
import gzip
import io
from pathlib import Path

from common import ROOT, check_hash, read_json, sha256, write_json


def load_rows(path):
    with Path(path).open(newline='') as stream:
        rows = list(csv.reader(stream))
    names = rows[0][1:]
    records = {row[0]: row[1:] for row in rows[1:]}
    if len(records) != len(rows) - 1 or len(names) != 40:
        raise ValueError('Duplicate paths or invalid attribute vocabulary')
    for path, values in records.items():
        if Path(path).is_absolute() or '..' in Path(path).parts:
            raise ValueError(f'Unsafe image path: {path}')
        if len(values) != 40:
            raise ValueError(f'Invalid annotation width: {path}')
    return names, records


def csv_bytes(names, paths, rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['image'] + names)
    for path in paths:
        writer.writerow([path] + [int(float(v)) for v in rows[path]])
    return stream.getvalue().encode()


def prepare(official_train, official_val, images_root=None):
    hashes = read_json(ROOT / 'reference/input_hashes.json')
    check_hash(official_train, hashes['official_train']['sha256'])
    check_hash(official_val, hashes['official_val']['sha256'])
    names, train = load_rows(official_train)
    val_names, val = load_rows(official_val)
    if names != val_names or train.keys() & val.keys():
        raise ValueError('Organizer train/validation overlap or different label order')
    rows = train | val
    report = read_json(ROOT / 'splits/report.json')
    partitions = {}
    content = {}
    for split in ['train', 'tune', 'audit', 'val', 'teacher']:
        with gzip.open(ROOT / f'splits/{split}.txt.gz', 'rt') as stream:
            paths = stream.read().splitlines()
        if len(set(paths)) != len(paths) or not set(paths) <= rows.keys():
            raise ValueError(f'Duplicate or unknown paths in {split}')
        partitions[split] = set(paths)
        payload = csv_bytes(names, paths, rows)
        import hashlib
        expected = (hashes['teacher_csv']['sha256'] if split == 'teacher'
                    else report['partitions'][split]['sha256'])
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f'Reconstructed CSV differs from reference: {split}')
        content[split] = payload
    if partitions['train'] & partitions['val'] or partitions['tune'] & partitions['audit']:
        raise ValueError('Split overlap')
    if partitions['train'] | partitions['val'] != rows.keys():
        raise ValueError('Split coverage mismatch')
    if partitions['val'] != partitions['tune'] | partitions['audit']:
        raise ValueError('Holdout coverage mismatch')
    if not partitions['teacher'] <= partitions['train']:
        raise ValueError('Teacher rows must be in the weight-training split')
    with gzip.open(ROOT / 'splits/groups.json.gz', 'rt') as stream:
        import json
        groups = json.load(stream)
    group_sets = {s: {groups[p] for p in partitions[s]} for s in ['train', 'tune', 'audit']}
    if any(group_sets[a] & group_sets[b] for a, b in [('train','tune'), ('train','audit'), ('tune','audit')]):
        raise ValueError('Recorded identity/duplicate groups cross split boundaries')
    for split, payload in content.items():
        target = (ROOT / 'data/teacher.csv' if split == 'teacher'
                  else ROOT / f'data/splits/train90_v1/{split}.csv')
        if target.exists() and target.read_bytes() != payload:
            raise FileExistsError(f'Refusing to replace different annotations: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    if images_root is not None:
        images_root = Path(images_root).resolve()
        for source in ['PETA', 'Market1501', 'PA100k']:
            origin = images_root / source
            if not origin.is_dir():
                raise FileNotFoundError(f'Expected organizer image directory: {origin}')
            target = ROOT / 'data' / source
            if target.is_symlink() or target.exists():
                if target.resolve() != origin:
                    raise FileExistsError(f'Existing image directory points elsewhere: {target}')
            else:
                target.symlink_to(origin, target_is_directory=True)
        import sys
        sys.path.insert(0, str(ROOT / 'scripts'))
        from data import resolve_image
        for path in rows:
            resolve_image(path, ROOT / 'data')
    result = {'status': 'PASS', 'rows': {s: len(p) for s,p in partitions.items()},
              'all_split_csv_hashes_match': True, 'recorded_groups_disjoint': True,
              'images_checked': images_root is not None}
    write_json(ROOT / 'reports/local/data.json', result)
    print(result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--official-train', required=True, type=Path)
    p.add_argument('--official-val', required=True, type=Path)
    p.add_argument('--images-root', type=Path)
    a = p.parse_args()
    prepare(a.official_train, a.official_val, a.images_root)
