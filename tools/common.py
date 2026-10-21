"""Small dependency-free helpers for the reproducible release workflow."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def check_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f'SHA256 mismatch for {path}: expected {expected}, got {actual}')
    return actual
